"""
Inputs a runtime delivers to a node. A node handles one event at a time.
"""

from __future__ import annotations

from dataclasses import dataclass

from torchsched.types.actions import TimerKey
from torchsched.types.messages import Message
from torchsched.types.models import Job, NodeId, ReservationId


@dataclass(frozen=True, slots=True)
class Boot:
    """
    The node has started (or restarted, with fresh state).

    Components arm their initial timers here, for example the first gossip
    round or the SWIM protocol period.
    """


@dataclass(frozen=True, slots=True)
class Received:
    sender: NodeId
    msg: Message


@dataclass(frozen=True, slots=True)
class TimerFired:
    key: TimerKey


@dataclass(frozen=True, slots=True)
class JobSubmitted:
    """
    A client submitted a job at this node.
    """

    job: Job


@dataclass(frozen=True, slots=True)
class ExecutionFinished:
    """
    The runtime finished (or failed) work it was asked to `Execute`.
    """

    reservation_id: ReservationId
    succeeded: bool
    reason: str | None = None


Event = Boot | Received | TimerFired | JobSubmitted | ExecutionFinished
