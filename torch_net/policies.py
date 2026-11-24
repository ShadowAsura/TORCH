"""Which worker states to query and how to rank feasible placements."""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from random import Random
from typing import Protocol

from torch_net.models import JobRequest


@dataclass(frozen=True)
class WorkerEndpoint:
    name: str
    url: str


class DiscoveryPolicy(Protocol):
    name: str

    def select(
        self,
        workers: Sequence[WorkerEndpoint],
        *,
        sample_size: int,
        seed: int,
        job_id: str,
        attempt: int,
    ) -> list[WorkerEndpoint]: ...


class Centralized:
    name = "centralized"

    def select(self, workers, *, sample_size, seed, job_id, attempt):
        return list(workers)


class Sampling:
    name = "sampling"

    def select(self, workers, *, sample_size, seed, job_id, attempt):
        # A per-job seed makes comparisons repeatable without shared RNG state.
        rng = Random(f"{seed}:{job_id}:{attempt}")
        return rng.sample(list(workers), min(sample_size, len(workers)))


POLICIES: dict[str, DiscoveryPolicy] = {
    "centralized": Centralized(),
    "sampling": Sampling(),
}


def choose_workers(
    policy: str,
    workers: Sequence[WorkerEndpoint],
    *,
    sample_size: int,
    seed: int,
    job_id: str,
    attempt: int,
) -> list[WorkerEndpoint]:
    return POLICIES[policy].select(
        workers,
        sample_size=sample_size,
        seed=seed,
        job_id=job_id,
        attempt=attempt,
    )


def valid_resources(resources: object) -> bool:
    """Only rank complete, finite, internally consistent worker snapshots."""
    if not isinstance(resources, Mapping):
        return False
    try:
        total = resources["gpu_count"]
        free = resources["free_gpus"]
        allocated = resources["allocated_gpus"]
        vram = resources["gpu_vram_gb"]
        performance = resources["performance"]
        gpu_type = resources["gpu_type"]
        return (
            type(gpu_type) is str
            and bool(gpu_type.strip())
            and all(type(value) is int for value in (total, free, allocated))
            and total > 0
            and 0 <= free <= total
            and 0 <= allocated <= total
            and free + allocated == total
            and type(vram) in (int, float)
            and math.isfinite(vram)
            and vram > 0
            and type(performance) in (int, float)
            and math.isfinite(performance)
            and performance > 0
        )
    except (KeyError, TypeError, ValueError, OverflowError):
        return False


def is_compatible(resources: Mapping, job: JobRequest) -> bool:
    """Discovery is advisory; only the worker may grant a reservation."""
    return (
        valid_resources(resources)
        and (
            not job.compatible_gpu_types
            or resources["gpu_type"] in job.compatible_gpu_types
        )
        and resources["free_gpus"] >= job.gpu_count
        and resources["gpu_vram_gb"] >= job.vram_gb
    )


def rank_compatible(
    snapshots: Sequence[tuple[WorkerEndpoint, Mapping]],
    job: JobRequest,
) -> list[tuple[WorkerEndpoint, Mapping]]:
    """Both discovery policies use the same placement objective.

    Prefer lower predicted simulated runtime, then less utilized capacity,
    then a stable worker ID. The worker still enforces the actual allocation.
    """
    eligible = [
        (endpoint, state) for endpoint, state in snapshots if is_compatible(state, job)
    ]
    return sorted(
        eligible,
        key=lambda item: (
            job.work_units / float(item[1]["performance"]),
            int(item[1]["allocated_gpus"]) / max(1, int(item[1]["gpu_count"])),
            item[0].name,
        ),
    )
