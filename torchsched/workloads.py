"""
Synthetic, seeded job workloads.
"""

from __future__ import annotations

from dataclasses import dataclass
from random import Random

from torchsched.types.models import Job

# (GPU count, minimum VRAM per GPU in GB, shortest and longest L40 runtime)
PROFILES = ((1, 24, 2, 6), (1, 64, 4, 10), (2, 48, 6, 14))
A100_SPEEDUP = 0.75  # an A100 runs a job in this fraction of its L40 time


@dataclass(frozen=True, slots=True)
class Submission:
    """
    A job and the time a client submits it.
    """

    at: float
    job: Job


def poisson_workload(seed: int, count: int, arrival_rate: float) -> list[Submission]:
    """
    `count` jobs with exponential interarrival times (`arrival_rate` jobs
    per second), each drawn from `PROFILES` with a uniform L40 runtime.

    For a given seed this generates exactly the same jobs as the legacy
    `experiment.make_workload`, so results stay comparable with it.
    """
    if count < 0:
        raise ValueError("count must be nonnegative")
    if arrival_rate <= 0:
        raise ValueError("arrival_rate must be positive")
    rng = Random(seed)
    at = 0.0
    submissions = []
    for index in range(count):
        at += rng.expovariate(arrival_rate)
        gpu_count, min_vram, shortest, longest = rng.choice(PROFILES)
        runtime = rng.uniform(shortest, longest)
        job = Job(
            f"job-{index + 1}",
            gpu_count,
            min_vram,
            {"L40": runtime, "A100": runtime * A100_SPEEDUP},
        )
        submissions.append(Submission(at, job))
    return submissions
