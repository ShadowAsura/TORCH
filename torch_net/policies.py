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




class Centralized:
    name = "centralized"

    def select(self, workers, *, sample_size, seed, job_id, attempt):
        return list(workers)




POLICIES = {
    "centralized": Centralized(),
}






