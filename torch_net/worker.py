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




    async def reserve(self, request: JobRequest) -> tuple[JobStatus, bool]:
        """Allocate GPUs once; same-ID same-payload retries return the first result."""
        async with self._lock:
            self._counters["reservation_attempts"] += 1
            self._expire_locked(perf_counter())

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


