"""Worker-owned GPU accounting and asynchronous job execution.

Each worker instance represents one process. All transitions that consume or
release GPUs use the same asyncio lock; remote schedulers cannot overbook it.
"""

import asyncio
import math
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from time import perf_counter
from uuid import uuid4

from .models import JobRequest, JobState, JobStatus









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




