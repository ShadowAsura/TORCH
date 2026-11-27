"""
Combine components into one node that a runtime drives.
"""

from __future__ import annotations

from collections.abc import Callable
from random import Random

from torchsched.interfaces.component import (
    Component,
    DisseminationRole,
    MembershipRole,
    SchedulerRole,
    WorkerRole,
)
from torchsched.types.actions import Action, CancelTimer, SetTimer
from torchsched.types.events import Boot, Event, Received, TimerFired
from torchsched.types.models import NodeId, NodeSpec, Status


class NodeContext:
    """
    Read-only facts and queries available to every component of a node.

    Rule: a component may *read* another component's state through this
    context, but may only *change* it by sending a message (to itself, if the
    other component is on the same node). Reads are synchronous; changes go
    through the runtime, so local and remote reservations behave the same.
    """

    def __init__(self, spec: NodeSpec, incarnation: int, seed: int):
        self.spec = spec
        self.incarnation = incarnation
        self._seed = seed
        self._rngs: dict[str, Random] = {}
        self._neighbors: Callable[[], frozenset[NodeId]] | None = None
        self._local_status: Callable[[float], Status | None] | None = None

    @property
    def node_id(self) -> NodeId:
        return self.spec.node_id

    def rng(self, purpose: str) -> Random:
        """
        A separate deterministic stream for each purpose.

        Each purpose (for example "gossip-peer" or "sample") gets its own
        stream, so new random draws in one component do not change what
        another component draws. That keeps policy comparisons on the same
        seed meaningful.
        """
        if purpose not in self._rngs:
            self._rngs[purpose] = Random(
                f"{self._seed}:{self.node_id}:{self.incarnation}:{purpose}"
            )
        return self._rngs[purpose]

    def neighbors(self) -> frozenset[NodeId]:
        return _bound(self._neighbors, "neighbors")()

    def local_status(self, now: float) -> Status | None:
        """
        This node's own worker status, or None for a scheduler-only node.
        """
        return self._local_status(now) if self._local_status else None


def _bound(fn, name: str):
    if fn is None:
        raise RuntimeError(f"NodeContext.{name} used before Node bound it")
    return fn


class Node:
    """
    Route each event to the component that owns it.

    Every node has membership and dissemination. A node without GPU groups
    has no worker, and a node that never receives jobs (for example, a
    worker in the centralized baseline) has no scheduler.

    * Boot goes to every component, in construction order.
    * `Received(msg)` goes to the component whose `accepts` contains
      `type(msg)`. Other events are routed by their own type the same way.
    * `TimerFired` goes to the component named by the first element of the
      key, with that prefix removed.
    * Timer keys in SetTimer and CancelTimer are prefixed with the emitting
      component's name.
    """

    def __init__(
        self,
        ctx: NodeContext,
        *,
        membership: MembershipRole,
        dissemination: DisseminationRole,
        scheduler: SchedulerRole | None = None,
        worker: WorkerRole | None = None,
    ):
        self.ctx = ctx
        self.worker = worker
        self.membership = membership
        self.dissemination = dissemination
        self.scheduler = scheduler
        self.components: tuple[Component, ...] = tuple(
            c for c in (worker, membership, dissemination, scheduler) if c is not None
        )

        self._by_name: dict[str, Component] = {}
        self._routes: dict[type, Component] = {}
        for component in self.components:
            if component.name in self._by_name:
                raise ValueError(f"duplicate component name {component.name!r}")
            self._by_name[component.name] = component
            for kind in component.accepts:
                if kind in self._routes:
                    raise ValueError(
                        f"{kind.__name__} accepted by both {self._routes[kind].name!r} "
                        f"and {component.name!r}"
                    )
                self._routes[kind] = component

        ctx._neighbors = membership.neighbors
        ctx._local_status = worker.status if worker is not None else None

    @property
    def node_id(self) -> NodeId:
        return self.ctx.node_id

    def handle(self, event: Event, now: float) -> list[Action]:
        if isinstance(event, Boot):
            actions: list[Action] = []
            for component in self.components:
                actions.extend(self._wrap(component, component.handle(event, now)))
            return actions
        if isinstance(event, TimerFired):
            owner, *rest = event.key
            component = self._by_name[owner]
            return self._wrap(component, component.handle(TimerFired(tuple(rest)), now))
        kind = type(event.msg) if isinstance(event, Received) else type(event)
        component = self._routes.get(kind)
        if component is None:
            # Unknown messages are dropped, as a real node would drop them.
            # The runtime may log them.
            return []
        return self._wrap(component, component.handle(event, now))

    @staticmethod
    def _wrap(component: Component, actions: list[Action]) -> list[Action]:
        out: list[Action] = []
        for action in actions:
            if isinstance(action, SetTimer):
                action = SetTimer((component.name, *action.key), action.delay)
            elif isinstance(action, CancelTimer):
                action = CancelTimer((component.name, *action.key))
            out.append(action)
        return out
