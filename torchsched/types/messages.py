"""
Messages exchanged between nodes, and between components of one node.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from torchsched.types.models import (
    GroupIndex,
    Job,
    JobId,
    NodeId,
    ReservationId,
    Status,
)


class Message:
    """
    Base class. Subclasses are registered by name for the wire codec.
    """

    __slots__ = ()
    registry: ClassVar[dict[str, type[Message]]] = {}

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        # @dataclass(slots=True) re-creates the class, so the same class is
        # registered twice; only a same-named class elsewhere is a real clash.
        previous = Message.registry.get(cls.__name__)
        if previous is not None and previous.__qualname__ + previous.__module__ != (
            cls.__qualname__ + cls.__module__
        ):
            raise TypeError(f"duplicate message type {cls.__name__}")
        Message.registry[cls.__name__] = cls


# ==================================================
# Worker protocol: scheduler <-> worker
# ==================================================
# Placement takes two phases. A reservation holds GPUs under a lease that the
# worker enforces, and a start turns the reservation into running work. If
# a scheduler disappears between the two, its reservation expires instead of
# holding GPUs forever. The split also leaves room for multi-node jobs later.
# `request_id` makes retries idempotent on unreliable networks.


@dataclass(frozen=True, slots=True)
class ReserveRequest(Message):
    request_id: str
    job: Job
    # The GPU group the placement policy chose. None lets the worker pick any
    # group that fits
    group: GroupIndex | None = None


@dataclass(frozen=True, slots=True)
class ReserveReply(Message):
    request_id: str
    job_id: JobId
    granted: bool
    reservation_id: ReservationId | None = None
    group: GroupIndex | None = None  # the group actually reserved, when granted
    reason: str | None = None  # set when not granted
    # The worker's `Status.order_key()` right after the grant. A scheduler
    # uses it to tell whether a status record already includes this grant.
    order_key: tuple[int, int] | None = None
    # Set when not granted: the worker's current state, so the scheduler can
    # correct a stale view instead of retrying against it. Grants leave this
    # unset; their `order_key` is what assumptions need.
    status: Status | None = None


@dataclass(frozen=True, slots=True)
class StartRequest(Message):
    reservation_id: ReservationId


@dataclass(frozen=True, slots=True)
class StartReply(Message):
    reservation_id: ReservationId
    started: bool
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class JobFinished(Message):
    """
    Worker -> the scheduler that started the job.
    """

    job_id: JobId
    reservation_id: ReservationId
    succeeded: bool
    # The worker's `Status.order_key()` right after releasing the GPUs.
    order_key: tuple[int, int]
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class ProbeRequest(Message):
    """
    Sampling-time state query (Sparrow-style).
    """

    request_id: str


@dataclass(frozen=True, slots=True)
class ProbeReply(Message):
    request_id: str
    status: Status


# ==================================================
# State dissemination: node <-> node
# ==================================================


@dataclass(frozen=True, slots=True)
class StatusPush(Message):
    """
    Status records pushed to a peer.

    Gossip sends the sender's own record plus records it has learned from
    others. Bounded-staleness periodic updates send only the sender's own
    record. Receivers keep the record with the highest
    `(incarnation, version)` and drop records older than their staleness
    limit.
    """

    statuses: tuple[Status, ...]


# ==================================================
# Local notifications: between components of one node
# ==================================================


@dataclass(frozen=True, slots=True)
class StatusChanged(Message):
    """
    Worker -> dissemination: this node's free GPUs changed.
    """

    status: Status


@dataclass(frozen=True, slots=True)
class CapacityAvailable(Message):
    """
    Dissemination -> scheduler: the view gained free capacity (a new node,
    a restarted node, or more free GPUs), so another placement round may
    succeed.
    """


# ==================================================
# Discovery: scheduler <-> dissemination on the same node
# ==================================================
# Discovery may need network round trips (sampling), so it is a request and
# reply exchanged with the local dissemination component, not a direct call.
# A view-based policy replies immediately; sampling replies after its probes
# return or time out.


@dataclass(frozen=True, slots=True)
class LookupRequest(Message):
    lookup_id: str


@dataclass(frozen=True, slots=True)
class LookupReply(Message):
    lookup_id: str
    candidates: tuple[Status, ...]


# ==================================================
# Scheduling: scheduler <-> scheduler
# ==================================================


@dataclass(frozen=True, slots=True)
class ForwardJob(Message):
    """
    Hand ownership of a job to another scheduler (ring/small-world routing).
    """

    job: Job
    visited: tuple[NodeId, ...]  # schedulers that already tried this job, in order


# TODO: add SWIM membership messages (ping, ping-req, ack, join via introducers)
# when SWIM membership is implemented.
