"""
The component interface and the four roles a node is built from.
"""

from __future__ import annotations

from typing import ClassVar, Protocol

from torchsched.types.actions import Action
from torchsched.types.events import Event
from torchsched.types.models import NodeId, Status


class Component(Protocol):
    """
    A sans-I/O state machine.

    `name` namespaces the component's timers. `accepts` lists the
    message types and the event types (JobSubmitted, ExecutionFinished) that
    Node routes to it. Every component receives Boot. A component receives
    TimerFired only for timers it armed, with its own key (no prefix).
    """

    name: ClassVar[str]
    accepts: ClassVar[frozenset[type]]

    def handle(self, event: Event, now: float) -> list[Action]: ...


class WorkerRole(Component, Protocol):
    """
    Owns this node's GPU groups: reservations, leases, executions.
    """

    def status(self, now: float) -> Status: ...


class MembershipRole(Component, Protocol):
    """
    Tracks which nodes exist and this node's neighborhood.

    TODO: handle SWIM (failure detection, join via introducers). For now
    membership is a static list. With SWIM, neighborhoods must be computed
    from the live member set, starting from the layouts in
    `torchsched.topology`. A shared shuffle like `ring_with_shortcuts`
    needs every node to agree on the member list, so churn needs a
    per-node rule.
    """

    def neighbors(self) -> frozenset[NodeId]: ...


class DisseminationRole(Component, Protocol):
    """
    Keeps a view of peers' state and answers LookupRequest.

    Implementations: push-on-change to a coordinator (centralized baseline),
    gossip, periodic bounded-staleness, and sampling (no standing view).
    """


class SchedulerRole(Component, Protocol):
    """
    Owns submitted jobs: queues, places, forwards, retries, and reports.
    """
