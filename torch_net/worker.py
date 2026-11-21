"""Worker-owned GPU accounting and asynchronous job execution.

Each worker instance represents one process. All transitions that consume or
release GPUs use the same asyncio lock; remote schedulers cannot overbook it.
"""

import asyncio
import logging
import math
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from time import perf_counter
from uuid import uuid4

from .models import JobRequest, JobState, JobStatus

log = logging.getLogger(__name__)


class ReservationConflict(Exception):
    """The worker cannot safely grant or start a requested reservation."""


class JobNotFound(Exception):
    """No reservation exists for this job ID on this worker."""


async def _wait_until(deadline: float) -> None:
    """Guard against asyncio timers waking early on coarse Windows clocks."""
    while (remaining := deadline - perf_counter()) > 0:
        await asyncio.sleep(max(remaining, 0.001))


@dataclass(frozen=True, slots=True)
class WorkerSettings:
    worker_id: str = "worker-1"
    gpu_type: str = "L40"
    gpu_count: int = 2
    gpu_vram_gb: float = 48.0
    gpu_performance: float = 1.0
    sim_seconds_per_work_unit: float = 1.0
    reservation_lease_seconds: float = 30.0

    def __post_init__(self) -> None:
        if not self.worker_id.strip() or not self.gpu_type.strip():
            raise ValueError("WORKER_ID and GPU_TYPE must be nonempty")
        if isinstance(self.gpu_count, bool) or self.gpu_count < 1:
            raise ValueError("GPU_COUNT must be a positive integer")
        for name, value in (
            ("GPU_VRAM_GB", self.gpu_vram_gb),
            ("GPU_PERFORMANCE", self.gpu_performance),
            ("SIM_SECONDS_PER_WORK_UNIT", self.sim_seconds_per_work_unit),
            ("RESERVATION_LEASE_SECONDS", self.reservation_lease_seconds),
        ):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be a finite positive number")

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> "WorkerSettings":
        env = os.environ if environ is None else environ
        return cls(
            worker_id=env.get("WORKER_ID", "worker-1"),
            gpu_type=env.get("GPU_TYPE", "L40"),
            gpu_count=int(env.get("GPU_COUNT", "2")),
            gpu_vram_gb=float(env.get("GPU_VRAM_GB", "48")),
            gpu_performance=float(env.get("GPU_PERFORMANCE", "1")),
            sim_seconds_per_work_unit=float(env.get("SIM_SECONDS_PER_WORK_UNIT", "1")),
            reservation_lease_seconds=float(env.get("RESERVATION_LEASE_SECONDS", "30")),
        )


@dataclass(slots=True)
class _JobRecord:
    request: JobRequest
    reservation_id: str
    reserved_at: float
    state: JobState = "reserved"
    error: str | None = None
    started_at: float | None = None
    allocated: bool = True
    lease_task: asyncio.Task[None] | None = field(default=None, repr=False)
    execution_task: asyncio.Task[None] | None = field(default=None, repr=False)


class Worker:
    def __init__(self, settings: WorkerSettings):
        self.settings = settings
        self._lock = asyncio.Lock()
        self._jobs: dict[str, _JobRecord] = {}
        self._allocated_gpus = 0
        self._closed = False
        self._created_at = perf_counter()
        self._busy_gpu_seconds = 0.0
        self._counters = {
            "reservation_attempts": 0,
            "reservations_granted": 0,
            "reservation_conflicts": 0,
            "reservation_duplicates": 0,
            "reservations_expired": 0,
            "jobs_started": 0,
            "jobs_completed": 0,
            "jobs_failed": 0,
        }

    def _snapshot(self, record: _JobRecord) -> JobStatus:
        return JobStatus(
            job_id=record.request.job_id,
            worker_id=self.settings.worker_id,
            reservation_id=record.reservation_id,
            state=record.state,
            gpu_count=record.request.gpu_count,
            error=record.error,
        )

    def _release_locked(self, record: _JobRecord) -> None:
        if record.allocated:
            self._allocated_gpus -= record.request.gpu_count
            record.allocated = False

    def _expire_locked(self, now: float) -> None:
        for record in self._jobs.values():
            if (
                record.state == "reserved"
                and now - record.reserved_at >= self.settings.reservation_lease_seconds
            ):
                record.state = "failed"
                record.error = "reservation_expired"
                self._release_locked(record)
                self._counters["reservations_expired"] += 1
                self._counters["jobs_failed"] += 1
                if (
                    record.lease_task is not asyncio.current_task()
                    and record.lease_task is not None
                ):
                    record.lease_task.cancel()

    async def _lease_timeout(
        self, job_id: str, reservation_id: str, deadline: float
    ) -> None:
        try:
            await _wait_until(deadline)
            async with self._lock:
                record = self._jobs.get(job_id)
                if record is not None and record.reservation_id == reservation_id:
                    self._expire_locked(perf_counter())
        except asyncio.CancelledError:
            return

    async def reserve(self, request: JobRequest) -> tuple[JobStatus, bool]:
        """Allocate GPUs once; same-ID same-payload retries return the first result."""
        async with self._lock:
            self._counters["reservation_attempts"] += 1
            self._expire_locked(perf_counter())
            existing = self._jobs.get(request.job_id)
            if existing is not None:
                if existing.request != request:
                    self._counters["reservation_conflicts"] += 1
                    raise ReservationConflict(
                        "job_id_already_used_with_different_request"
                    )
                self._counters["reservation_duplicates"] += 1
                return self._snapshot(existing), False

            if (
                self.settings.gpu_type not in request.compatible_gpu_types
                and request.compatible_gpu_types
            ):
                error = "incompatible_gpu_type"
            elif request.vram_gb > self.settings.gpu_vram_gb:
                error = "insufficient_gpu_vram"
            elif request.gpu_count > self.settings.gpu_count - self._allocated_gpus:
                error = "insufficient_free_gpus"
            elif not math.isfinite(
                request.work_units
                / self.settings.gpu_performance
                * self.settings.sim_seconds_per_work_unit
            ):
                error = "invalid_simulated_duration"
            else:
                error = None
            if error is not None:
                self._counters["reservation_conflicts"] += 1
                raise ReservationConflict(error)

            record = _JobRecord(
                request=request.model_copy(deep=True),
                reservation_id=str(uuid4()),
                reserved_at=perf_counter(),
            )
            self._jobs[request.job_id] = record
            self._allocated_gpus += request.gpu_count
            self._counters["reservations_granted"] += 1
            log.debug("Reserved %s on %s", request.job_id, self.settings.worker_id)
            record.lease_task = asyncio.create_task(
                self._lease_timeout(
                    request.job_id,
                    record.reservation_id,
                    record.reserved_at + self.settings.reservation_lease_seconds,
                )
            )
            return self._snapshot(record), True

    async def start(self, job_id: str) -> JobStatus:
        async with self._lock:
            self._expire_locked(perf_counter())
            record = self._jobs.get(job_id)
            if record is None:
                raise JobNotFound(job_id)
            if record.state == "failed":
                raise ReservationConflict(record.error or "job_failed")
            if record.state == "reserved":
                record.state = "running"
                record.started_at = perf_counter()
                if record.lease_task is not None:
                    record.lease_task.cancel()
                self._counters["jobs_started"] += 1
                record.execution_task = asyncio.create_task(self._execute(record))
            return self._snapshot(record)

    async def _execute(self, record: _JobRecord) -> None:
        """Release GPUs once on success, failure or cancellation."""
        error: str | None = None
        try:
            delay = (
                record.request.work_units
                / self.settings.gpu_performance
                * self.settings.sim_seconds_per_work_unit
            )
            assert record.started_at is not None
            await _wait_until(record.started_at + delay)
        except asyncio.CancelledError:
            error = "execution_cancelled"
        except Exception as exc:  # noqa: BLE001 - finally always returns GPU capacity
            error = f"execution_error:{type(exc).__name__}"
        finally:
            async with self._lock:
                if record.state == "running":
                    assert record.started_at is not None
                    self._busy_gpu_seconds += record.request.gpu_count * (
                        perf_counter() - record.started_at
                    )
                    if error is None:
                        record.state = "completed"
                        log.debug(
                            "Completed %s on %s",
                            record.request.job_id,
                            self.settings.worker_id,
                        )
                    else:
                        record.state = "failed"
                        record.error = error
                        log.warning("Job %s failed: %s", record.request.job_id, error)
                    self._release_locked(record)

    async def get_job(self, job_id: str) -> JobStatus:
        async with self._lock:
            self._expire_locked(perf_counter())
            record = self._jobs.get(job_id)
            if record is None:
                raise JobNotFound(job_id)
            return self._snapshot(record)

    async def resources(self) -> dict:
        async with self._lock:
            self._expire_locked(perf_counter())
            return {
                "worker_id": self.settings.worker_id,
                "gpu_type": self.settings.gpu_type,
                "gpu_count": self.settings.gpu_count,
                "free_gpus": self.settings.gpu_count - self._allocated_gpus,
                "gpu_vram_gb": self.settings.gpu_vram_gb,
                "available_vram_gb": (self.settings.gpu_count - self._allocated_gpus)
                * self.settings.gpu_vram_gb,
                "performance": self.settings.gpu_performance,
                "allocated_gpus": self._allocated_gpus,
                "active_jobs": sum(job.allocated for job in self._jobs.values()),
            }


    async def close(self) -> None:
        """Stop outstanding tasks and account for interrupted runs on shutdown."""
        async with self._lock:
            self._closed = True
            now = perf_counter()
            tasks = [
                task
                for record in self._jobs.values()
                for task in (record.lease_task, record.execution_task)
                if task is not None and not task.done()
            ]
            for task in tasks:
                task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
