"""
Outputs a node returns for its runtime to carry out.

A node never performs effects itself; it describes them as actions. The
runtime carries out a node's actions in the order the node returned them.
"""

from __future__ import annotations

from dataclasses import dataclass

from torchsched.types.messages import Message
from torchsched.types.models import GpuType, GroupIndex, Job, NodeId, ReservationId
from torchsched.types.metrics import Metric

# Inside a component, keys are local, for example ("gossip",) or
# ("lease", "job-7"). Node prefixes each key with the component's name, so
# components never collide and never see each other's timers.
TimerKey = tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Send:
    """
    Deliver ``msg`` to node ``to``, which receives it as ``Received(sender, msg)``.

    Delivery may be delayed or lost, so protocols must tolerate both. A
    message a node sends to itself is delivered at the same time, after
    events already queued for that time, and is not network traffic.
    """

    to: NodeId
    msg: Message


@dataclass(frozen=True, slots=True)
class SetTimer:
    """
    Deliver ``TimerFired(key)`` to this node after ``delay`` seconds.

    Re-arming an existing key replaces the pending timer.
    """

    key: TimerKey
    delay: float

    def __post_init__(self) -> None:
        if self.delay < 0:
            raise ValueError("timer delay must be nonnegative")


@dataclass(frozen=True, slots=True)
class CancelTimer:
    """
    Disarm a pending timer so it never fires. Cancelling a timer that is not
    armed does nothing.
    """

    key: TimerKey


@dataclass(frozen=True, slots=True)
class Execute:
    """
    Run ``job`` on GPUs reserved in ``group`` of this node.

    When the work ends, the node receives
    ``ExecutionFinished(reservation_id, succeeded, reason)``. ``gpu_type`` is
    the group's GPU type, so the job's expected runtime is
    ``job.runtime_by_gpu[gpu_type]``.
    """

    reservation_id: ReservationId
    job: Job
    group: GroupIndex
    gpu_type: GpuType


@dataclass(frozen=True, slots=True)
class Observe:
    """
    Record a metric. It has no effect on any node's behavior.

    Each recorded metric is stamped with the current time and the node
    that observed it.
    """

    metric: Metric


Action = Send | SetTimer | CancelTimer | Execute | Observe
