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


    def validate(self) -> None:
        if not self.workers or len({worker.name for worker in self.workers}) != len(
            self.workers
        ):
            raise SchedulingConfigurationError(
                "workers must be nonempty with unique names"
            )
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




