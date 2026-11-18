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






class Scheduler:
    """Owns job submissions for one process only; workers enforce capacity."""












