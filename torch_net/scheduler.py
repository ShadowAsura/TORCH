"""Networked job coordinator; worker processes own resource reservations."""

import asyncio
import math
import os
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from torch_net.models import JobRequest
from torch_net.policies import (
    POLICIES,
    WorkerEndpoint,
    choose_workers,
    rank_compatible,
    valid_resources,
)


class SchedulingRequest(JobRequest):
    policy: str | None = None


class SchedulingConfigurationError(ValueError):
    """Invalid settings should fail at startup rather than after submission."""


class SchedulerClosedError(RuntimeError):
    """Submissions are unavailable once the scheduler is shutting down."""


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
    monitor_timeout_seconds: float = 3600.0

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
            monitor_timeout_seconds=float(
                os.getenv("SCHEDULER_MONITOR_TIMEOUT_SECONDS", "3600")
            ),
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
        if (
            any(
                type(value) is not int
                for value in (
                    self.sample_size,
                    self.max_retries,
                    self.max_monitor_errors,
                    self.seed,
                )
            )
            or self.sample_size < 1
            or self.max_retries < 0
            or self.max_monitor_errors < 1
        ):
            raise SchedulingConfigurationError("invalid sample size or retry limit")
        try:
            finite = all(
                type(value) in (int, float) and math.isfinite(value) and value > 0
                for value in (
                    self.retry_delay_seconds,
                    self.timeout_seconds,
                    self.poll_seconds,
                    self.monitor_timeout_seconds,
                )
            )
        except OverflowError:
            finite = False
        if not finite:
            raise SchedulingConfigurationError(
                "timeouts and retry delays must be finite and positive"
            )
        for worker in self.workers:
            if (
                not isinstance(worker.name, str)
                or not worker.name.strip()
                or worker.name != worker.name.strip()
            ):
                raise SchedulingConfigurationError("worker names must be nonempty")
            try:
                if not isinstance(worker.url, str):
                    raise SchedulingConfigurationError("worker URLs must be strings")
                url = httpx.URL(worker.url)
                if (
                    url.scheme not in ("http", "https")
                    or not url.host
                    or url.query
                    or url.fragment
                    or url.username
                    or url.password
                    or url.path != "/"
                    or worker.url.endswith("/")
                ):
                    raise SchedulingConfigurationError(
                        f"invalid worker URL for {worker.name!r}"
                    )
            except (httpx.InvalidURL, TypeError, ValueError) as exc:
                raise SchedulingConfigurationError(
                    f"invalid worker URL for {worker.name!r}"
                ) from exc


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
        self._closed = False
        self._reservation_successes = 0
        self._reservation_rejections = 0
        self._worker_errors = 0

    async def close(self) -> None:
        async with self._lock:
            self._closed = True
        if self._tasks:
            for task in self._tasks:
                task.cancel()
            await asyncio.gather(*self._tasks, return_exceptions=True)
        for record in self._jobs.values():
            if record.state not in ("completed", "failed"):
                record.state = "failed"
                record.reason = "scheduler stopped; worker outcome unknown"
                if record.scheduling_latency_ms is None:
                    record.scheduling_latency_ms = round(
                        (time.perf_counter() - record.accepted_at) * 1000, 3
                    )
        if self._owns_client:
            await self.client.aclose()

    def _create_task(self, coroutine) -> None:
        task = asyncio.create_task(coroutine)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def submit(self, job: SchedulingRequest) -> tuple[dict, bool]:
        policy = self.settings.policy if job.policy is None else job.policy
        if policy not in POLICIES:
            raise SchedulingConfigurationError(f"unknown policy: {policy}")
        async with self._lock:
            if self._closed:
                raise SchedulerClosedError("scheduler is closed")
            existing = self._jobs.get(job.job_id)
            if existing is not None:
                if existing.request != job:
                    raise ValueError("job_id already used for another request")
                return existing.view(), False
            record = JobRecord(request=job, policy=policy)
            self._jobs[job.job_id] = record
            self._create_task(self._run(record))
            return record.view(), True

    def status(self, job_id: str) -> dict | None:
        record = self._jobs.get(job_id)
        return record.view() if record else None

    def metrics(self) -> dict[str, Any]:
        records = list(self._jobs.values())
        delays = [
            r.scheduling_latency_ms
            for r in records
            if r.started_at is not None and r.scheduling_latency_ms is not None
        ]
        durations = [
            r.execution_seconds for r in records if r.execution_seconds is not None
        ]
        return {
            "jobs_submitted": len(records),
            "jobs_completed": sum(r.state == "completed" for r in records),
            "jobs_failed": sum(r.state == "failed" for r in records),
            "worker_state_queries": sum(r.query_count for r in records),
            "reservation_successes": self._reservation_successes,
            "reservation_rejections": self._reservation_rejections,
            "scheduling_retries": sum(r.retries for r in records),
            "worker_network_errors": self._worker_errors,
            "mean_scheduling_latency_ms": sum(delays) / len(delays) if delays else None,
            "mean_execution_seconds": sum(durations) / len(durations)
            if durations
            else None,
            "note": "Application message bytes are deferred; no estimated byte counts are reported.",
        }

    async def _query(self, endpoint: WorkerEndpoint, record: JobRecord):
        record.query_count += 1
        record.queried_workers.append(endpoint.name)
        try:
            response = await self.client.get(f"{endpoint.url}/resources")
            response.raise_for_status()
            resources = response.json()
            if (
                not valid_resources(resources)
                or resources.get("worker_id") != endpoint.name
            ):
                raise ValueError(f"invalid resource snapshot from {endpoint.name}")
            return endpoint, resources
        except (httpx.HTTPError, ValueError, TypeError):
            self._worker_errors += 1
            return None

    @staticmethod
    def _worker_job_state(
        data: object, endpoint: WorkerEndpoint, record: JobRecord
    ) -> str:
        """Reject status for a different job, worker, or reservation."""
        if not isinstance(data, dict):
            raise TypeError("worker job status must be a JSON object")
        reservation_id = data.get("reservation_id")
        state = data.get("state")
        if (
            data.get("job_id") != record.request.job_id
            or data.get("worker_id") != endpoint.name
            or type(data.get("gpu_count")) is not int
            or data["gpu_count"] != record.request.gpu_count
            or not isinstance(reservation_id, str)
            or not reservation_id
            or (
                record.reservation_id is not None
                and reservation_id != record.reservation_id
            )
            or state not in ("reserved", "running", "completed", "failed")
        ):
            raise ValueError(f"invalid job status from {endpoint.name}")
        return state

    async def _reserve(self, endpoint: WorkerEndpoint, record: JobRecord) -> str:
        request_body = record.request.model_dump(exclude={"policy"})
        try:
            response = await self.client.post(
                f"{endpoint.url}/reservations",
                json=request_body,
            )
            if response.status_code in (200, 201):
                try:
                    data = response.json()
                    state = self._worker_job_state(data, endpoint, record)
                    if state == "failed":
                        raise ValueError("worker reported a failed reservation")
                except (TypeError, ValueError):
                    # The worker may already hold GPUs. Reconcile before any retry.
                    self._worker_errors += 1
                    return "uncertain"
                record.reservation_id = data["reservation_id"]
                return "granted"
            if response.status_code in (409, 422):
                record.reservation_rejections += 1
                self._reservation_rejections += 1
                return "rejected"
            self._worker_errors += 1
            return "uncertain"
        except (httpx.HTTPError, ValueError, TypeError):
            self._worker_errors += 1
            return "uncertain"

    async def _reconcile(self, endpoint: WorkerEndpoint, record: JobRecord) -> str:
        """After an ambiguous POST, only accept proof from the same worker."""
        for attempt in range(self.settings.max_monitor_errors):
            try:
                response = await self.client.get(
                    f"{endpoint.url}/jobs/{record.request.job_id}"
                )
                response.raise_for_status()
                data = response.json()
                state = self._worker_job_state(data, endpoint, record)
                record.reservation_id = data["reservation_id"]
                return state
            except (httpx.HTTPError, ValueError, TypeError):
                self._worker_errors += 1
                if attempt + 1 < self.settings.max_monitor_errors:
                    await asyncio.sleep(self.settings.retry_delay_seconds)
        # A timeout could mean the worker has reserved resources even if the
        # subsequent lookup returns 404 while the request is still in flight.
        # Never dispatch the same logical job on another worker in this case.
        raise RuntimeError("reservation outcome unknown after worker connection error")

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
                self._worker_errors += 1
        except (httpx.HTTPError, ValueError):
            # Even an error reply can follow a successful remote start.
            self._worker_errors += 1

        errors = 0
        reserved_polls = 0
        deadline = record.started_at + self.settings.monitor_timeout_seconds
        while True:
            if time.perf_counter() >= deadline:
                record.reason = "worker monitoring timed out; outcome unknown"
                record.state = "failed"
                return
            await asyncio.sleep(self.settings.poll_seconds)
            try:
                response = await self.client.get(
                    f"{endpoint.url}/jobs/{record.request.job_id}"
                )
                response.raise_for_status()
                state = self._worker_job_state(response.json(), endpoint, record)
                errors = 0
                if state == "completed":
                    record.state = state
                    record.execution_seconds = round(
                        time.perf_counter() - record.started_at, 6
                    )
                    return
                if state == "failed":
                    record.reason = "worker reported execution failure"
                    record.state = state
                    return
                if state == "running":
                    record.state = state
                    reserved_polls = 0
                if state == "reserved":
                    record.state = state
                    reserved_polls += 1
                    if reserved_polls >= self.settings.max_monitor_errors:
                        record.reason = "worker never started; outcome unknown"
                        record.state = "failed"
                        return
                    # The worker's start API is idempotent. Retry only this worker.
                    start = await self.client.post(
                        f"{endpoint.url}/jobs/{record.request.job_id}/start"
                    )
                    if start.status_code not in (200, 201, 202):
                        self._worker_errors += 1
            except (httpx.HTTPError, ValueError, TypeError):
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
                candidates = rank_compatible(snapshots, record.request)
                for endpoint, _ in candidates:
                    result = await self._reserve(endpoint, record)
                    if result == "rejected":
                        continue
                    if result == "uncertain":
                        state = await self._reconcile(endpoint, record)
                        if state == "failed":
                            raise RuntimeError(
                                "worker reported failed reservation after ambiguous reply"
                            )
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
                if attempt < self.settings.max_retries:
                    record.retries += 1
                    await asyncio.sleep(self.settings.retry_delay_seconds)
            record.state = "failed"
            record.reason = record.reason or "no capacity after bounded retries"
            record.scheduling_latency_ms = round(
                (time.perf_counter() - record.accepted_at) * 1000, 3
            )
        except asyncio.CancelledError:
            record.state = "failed"
            record.reason = (
                "scheduler stopped; worker outcome unknown"
                if record.selected_worker
                else (
                    "scheduler stopped before placement; reservation outcome may be unknown"
                )
            )
            if record.scheduling_latency_ms is None:
                record.scheduling_latency_ms = round(
                    (time.perf_counter() - record.accepted_at) * 1000, 3
                )
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
        return (
            valid_resources(snapshot)
            and job.gpu_count <= snapshot["gpu_count"]
            and job.vram_gb <= snapshot["gpu_vram_gb"]
            and (
                not job.compatible_gpu_types
                or snapshot["gpu_type"] in job.compatible_gpu_types
            )
        )
