"""Which worker states to query and how to rank feasible placements."""

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


def is_compatible(resources: Mapping, job: JobRequest) -> bool:
    """Resources are authoritative only when the worker reserves atomically."""
    try:
        gpu_type = str(resources["gpu_type"])
        return (
            (not job.compatible_gpu_types or gpu_type in job.compatible_gpu_types)
            and int(resources["free_gpus"]) >= job.gpu_count
            and float(resources["gpu_vram_gb"]) >= job.vram_gb
            and float(resources["performance"]) > 0
        )
    except (TypeError, ValueError, KeyError):
        return False


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
