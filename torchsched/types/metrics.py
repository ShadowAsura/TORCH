"""
Observations components emit for metrics.
"""

from __future__ import annotations

from dataclasses import dataclass

from torchsched.types.models import GpuType, GroupIndex, JobId, NodeId


@dataclass(frozen=True, slots=True)
class JobAccepted:
    """
    A client's job entered the system at this node.
    """

    job_id: JobId


@dataclass(frozen=True, slots=True)
class JobForwarded:
    job_id: JobId
    to: NodeId


@dataclass(frozen=True, slots=True)
class ReservationRejected:
    job_id: JobId
    worker: NodeId
    reason: str


@dataclass(frozen=True, slots=True)
class PlacementRetried:
    """
    A placement round found nothing usable; the job will be tried again.
    """

    job_id: JobId
    reason: str


@dataclass(frozen=True, slots=True)
class RequestAbandoned:
    """
    A request for a job got no reply after every resend, so the scheduler
    gave up on it. `request` is the message type, for example
    "ReserveRequest".
    """

    job_id: JobId
    to: NodeId
    request: str


@dataclass(frozen=True, slots=True)
class JobStarted:
    job_id: JobId
    worker: NodeId
    group: GroupIndex
    gpu_type: GpuType
    gpu_count: int


@dataclass(frozen=True, slots=True)
class JobCompleted:
    job_id: JobId
    worker: NodeId


@dataclass(frozen=True, slots=True)
class JobInterrupted:
    """
    The scheduler believes the job's run was lost (its worker stopped
    sending heartbeats or restarted), so the job is queued again.
    """

    job_id: JobId
    worker: NodeId
    reason: str


@dataclass(frozen=True, slots=True)
class JobFailed:
    """
    Terminal failure: the job will not be retried.
    """

    job_id: JobId
    reason: str


Metric = (
    JobAccepted
    | JobForwarded
    | ReservationRejected
    | PlacementRetried
    | RequestAbandoned
    | JobStarted
    | JobCompleted
    | JobInterrupted
    | JobFailed
)
