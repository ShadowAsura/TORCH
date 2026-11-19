"""Networked job coordinator; worker processes own resource reservations."""

import asyncio
import os
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from torch_net.models import JobRequest
from torch_net.policies import POLICIES, WorkerEndpoint, choose_workers, rank_compatible


class SchedulingRequest(JobRequest):
    policy: str | None = None


class SchedulingConfigurationError(ValueError):
    """Invalid settings should fail at startup rather than after submission."""


@dataclass(frozen=True)
class SchedulerSettings:
    workers: tuple[WorkerEndpoint, ...]
    policy: str = "centralized"
    sample_size: int = 2
    seed: int = 42
    max_retries: int = 20
    retry_delay_seconds: float = 0.1
    timeout_seconds: float = 1.0
    poll_seconds: float = 0.05
    max_monitor_errors: int = 5

    @classmethod
    def from_env(cls) -> "SchedulerSettings":
        raw = os.getenv(
            "SCHEDULER_WORKERS",
            "worker-l40=http://localhost:8001,worker-a100=http://localhost:8002,"
            "worker-t4=http://localhost:8003",
        )
        workers = []
        for entry in raw.split(","):
            if not entry.strip():
                continue
            name, sep, url = entry.strip().partition("=")
            if not sep or not name.strip() or not url.strip():
                raise SchedulingConfigurationError(
                    "SCHEDULER_WORKERS requires name=http://host:port entries"
                )
            workers.append(WorkerEndpoint(name.strip(), url.strip().rstrip("/")))
        settings = cls(
            workers=tuple(workers),
            policy=os.getenv("SCHEDULER_POLICY", "centralized"),
            sample_size=int(os.getenv("SCHEDULER_SAMPLE_SIZE", "2")),
            seed=int(os.getenv("SCHEDULER_SEED", "42")),
            max_retries=int(os.getenv("SCHEDULER_MAX_RETRIES", "20")),
            retry_delay_seconds=float(
                os.getenv("SCHEDULER_RETRY_DELAY_SECONDS", "0.1")
            ),
            timeout_seconds=float(os.getenv("SCHEDULER_REQUEST_TIMEOUT_SECONDS", "1")),
            poll_seconds=float(os.getenv("SCHEDULER_POLL_SECONDS", "0.05")),
            max_monitor_errors=int(os.getenv("SCHEDULER_MAX_MONITOR_ERRORS", "5")),
        )
        settings.validate()
        return settings

    def validate(self) -> None:
        if not self.workers or len({worker.name for worker in self.workers}) != len(
            self.workers
        ):
            raise SchedulingConfigurationError(
                "workers must be nonempty with unique names"
            )
        if self.policy not in POLICIES:
            raise SchedulingConfigurationError(f"unknown policy: {self.policy}")
        if min(self.retry_delay_seconds, self.timeout_seconds, self.poll_seconds) <= 0:
            raise SchedulingConfigurationError(
                "timeouts and retry delays must be positive"
            )


@dataclass
class JobRecord:
    request: SchedulingRequest
    policy: str
    accepted_at: float = field(default_factory=time.perf_counter)
    state: str = "queued"
    selected_worker: str | None = None
    worker_url: str | None = None
    query_count: int = 0
    queried_workers: list[str] = field(default_factory=list)
    attempts: int = 0
    retries: int = 0
    reservation_rejections: int = 0
    scheduling_latency_ms: float | None = None
    execution_seconds: float | None = None
    started_at: float | None = None
    reason: str | None = None
    reservation_id: str | None = None

    def view(self) -> dict[str, Any]:
        return {
            "job_id": self.request.job_id,
            "state": self.state,
            "policy": self.policy,
            "selected_worker": self.selected_worker,
            "query_count": self.query_count,
            "queried_workers": list(self.queried_workers),
            "attempts": self.attempts,
            "retries": self.retries,
            "reservation_rejections": self.reservation_rejections,
            "scheduling_latency_ms": self.scheduling_latency_ms,
            "execution_seconds": self.execution_seconds,
            "reason": self.reason,
            "reservation_id": self.reservation_id,
        }


class Scheduler:
    """Owns job submissions for one process only; workers enforce capacity."""

    def __init__(
        self, settings: SchedulerSettings, *, client: httpx.AsyncClient | None = None
    ):
        settings.validate()
        self.settings = settings
        self.client = client or httpx.AsyncClient(timeout=settings.timeout_seconds)
        self._owns_client = client is None
        self._jobs: dict[str, JobRecord] = {}
        self._tasks: set[asyncio.Task] = set()
        self._lock = asyncio.Lock()
        self._reservation_successes = 0
        self._reservation_rejections = 0
        self._worker_errors = 0

    async def close(self) -> None:
        if self._tasks:
            for task in self._tasks:
                task.cancel()
            await asyncio.gather(*self._tasks, return_exceptions=True)
        if self._owns_client:
            await self.client.aclose()

    def _create_task(self, coroutine) -> None:
        task = asyncio.create_task(coroutine)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def submit(self, job: SchedulingRequest) -> tuple[dict, bool]:
        policy = job.policy or self.settings.policy
        if policy not in POLICIES:
            raise SchedulingConfigurationError(f"unknown policy: {policy}")
        async with self._lock:
            record = JobRecord(request=job, policy=policy)
            self._jobs[job.job_id] = record
            self._create_task(self._run(record))
            return record.view(), True



    async def _query(self, endpoint: WorkerEndpoint, record: JobRecord):
        try:
            response = await self.client.get(f"{endpoint.url}/resources")
            response.raise_for_status()
            return endpoint, response.json()
        except (httpx.HTTPError, ValueError):
            return None

    async def _reserve(self, endpoint: WorkerEndpoint, record: JobRecord) -> str:
        request_body = record.request.model_dump(exclude={"policy"})
        try:
            response = await self.client.post(
                f"{endpoint.url}/reservations",
                json=request_body,
            )
            if response.status_code in (200, 201):
                reservation_id = response.json().get("reservation_id")
                if not reservation_id:
                    # The resource may now be reserved even if the reply is invalid.
                    return "uncertain"
                record.reservation_id = reservation_id
                return "granted"
            self._worker_errors += 1
            return "uncertain"
        except (httpx.HTTPError, ValueError):
            self._worker_errors += 1
            return "uncertain"


    async def _start_and_monitor(
        self, endpoint: WorkerEndpoint, record: JobRecord
    ) -> None:
        # Once reserved, the scheduler NEVER dispatches this job elsewhere.
        record.selected_worker = endpoint.name
        record.worker_url = endpoint.url
        record.scheduling_latency_ms = round(
            (time.perf_counter() - record.accepted_at) * 1000, 3
        )
        record.state = "reserved"
        self._reservation_successes += 1
        # Include the initial start RPC in the observed job duration. A very
        # short simulated job can finish before the HTTP start reply arrives.
        record.started_at = time.perf_counter()
        try:
            response = await self.client.post(
                f"{endpoint.url}/jobs/{record.request.job_id}/start",
            )
            if response.status_code not in (200, 201, 202):
                raise RuntimeError(
                    f"worker refused to start job: HTTP {response.status_code}"
                )
            record.state = "running"
        except (httpx.HTTPError, ValueError):
            # Starting may have succeeded remotely. Reconcile by polling that
            # worker rather than submitting the job to another worker.
            record.state = "running"
        errors = 0
        while True:
            await asyncio.sleep(self.settings.poll_seconds)
            try:
                response = await self.client.get(
                    f"{endpoint.url}/jobs/{record.request.job_id}"
                )
                response.raise_for_status()
                state = response.json().get("state")
                errors = 0
                if state == "completed":
                    record.state = state
                    record.execution_seconds = round(
                        time.perf_counter() - record.started_at, 6
                    )
                    return
                if state == "reserved":
                    # Duplicate start is idempotent; retry after an ambiguous reply.
                    await self.client.post(
                        f"{endpoint.url}/jobs/{record.request.job_id}/start"
                    )
            except (httpx.HTTPError, ValueError):
                self._worker_errors += 1
                errors += 1
                if errors >= self.settings.max_monitor_errors:
                    record.reason = (
                        "worker unavailable after reservation; outcome unknown"
                    )
                    record.state = "failed"
                    return

    async def _run(self, record: JobRecord) -> None:
        try:
            for attempt in range(self.settings.max_retries + 1):
                record.attempts += 1
                chosen = choose_workers(
                    record.policy,
                    self.settings.workers,
                    sample_size=self.settings.sample_size,
                    seed=self.settings.seed,
                    job_id=record.request.job_id,
                    attempt=attempt,
                )
                replies = await asyncio.gather(
                    *(self._query(e, record) for e in chosen)
                )
                snapshots = [reply for reply in replies if reply is not None]
                candidates = sorted(
                    ((e, s) for e, s in snapshots if
                     int(s["free_gpus"]) >= record.request.gpu_count
                     and float(s["gpu_vram_gb"]) >= record.request.vram_gb
                     and (not record.request.compatible_gpu_types
                          or s["gpu_type"] in record.request.compatible_gpu_types)),
                    key=lambda pair: -float(pair[1]["performance"]),
                )
                for endpoint, _ in candidates:
                    result = await self._reserve(endpoint, record)
                    if result == "rejected":
                        continue
                    if result == "uncertain":
                        await self._reconcile(endpoint, record)
                    await self._start_and_monitor(endpoint, record)
                    return
                # Static incompatibility is known only when all configured
                # workers have replied, not when a sampled subset has replied.
                all_replied = len(snapshots) == len(self.settings.workers)
                if all_replied and not any(
                    self._could_fit(snapshot, record.request)
                    for _, snapshot in snapshots
                ):
                    record.reason = "no worker has compatible GPU capacity"
                    break
            record.state = "failed"
            record.reason = record.reason or "no capacity after bounded retries"
            record.scheduling_latency_ms = round(
                (time.perf_counter() - record.accepted_at) * 1000, 3
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - task boundary records failures
            record.state = "failed"
            record.reason = str(exc)
            if record.scheduling_latency_ms is None:
                record.scheduling_latency_ms = round(
                    (time.perf_counter() - record.accepted_at) * 1000, 3
                )

    @staticmethod
    def _could_fit(snapshot: dict, job: JobRequest) -> bool:
        try:
            return (
                job.gpu_count <= int(snapshot["gpu_count"])
                and job.vram_gb <= float(snapshot["gpu_vram_gb"])
                and (
                    not job.compatible_gpu_types
                    or snapshot["gpu_type"] in job.compatible_gpu_types
                )
            )
        except (TypeError, ValueError, KeyError):
            return False
