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




