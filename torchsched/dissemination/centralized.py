"""
Centralized dissemination: every worker reports to one coordinator.
"""

from __future__ import annotations

from typing import ClassVar

from torchsched.dissemination.capacity import gained_capacity
from torchsched.node import NodeContext
from torchsched.types.actions import Action, Send
from torchsched.types.events import Boot, Event, Received
from torchsched.types.messages import (
    CapacityAvailable,
    LookupReply,
    LookupRequest,
    StatusChanged,
    StatusPush,
)
from torchsched.types.models import NodeId, Status


class CentralizedDissemination:
    """
    Workers push their status to `coordinator` on boot and on every change.

    The coordinator keeps the latest record per node and answers lookups
    with all of them, in the order nodes were first heard from. Every other
    node only forwards its own changes. With no network delay, the
    coordinator's view is exact, which makes this the idealized baseline.
    """

    name: ClassVar[str] = "dissemination"
    accepts: ClassVar[frozenset[type]] = frozenset(
        {StatusChanged, StatusPush, LookupRequest}
    )

    def __init__(self, ctx: NodeContext, *, coordinator: NodeId):
        self.ctx = ctx
        self.coordinator = coordinator
        self.view: dict[NodeId, Status] = {}

    @property
    def is_coordinator(self) -> bool:
        return self.ctx.node_id == self.coordinator

    def handle(self, event: Event, now: float) -> list[Action]:
        match event:
            case Boot():
                status = self.ctx.local_status(now)
                return self._report(status) if status is not None else []
            case Received(_, StatusChanged(status)):
                return self._report(status)
            case Received(_, StatusPush(statuses)):
                return self._store(statuses)
            case Received(sender, LookupRequest(lookup_id)):
                reply = LookupReply(lookup_id, tuple(self.view.values()))
                return [Send(sender, reply)]
        return []

    def _report(self, status: Status) -> list[Action]:
        if self.is_coordinator:
            return self._store((status,))
        return [Send(self.coordinator, StatusPush((status,)))]

    def _store(self, statuses: tuple[Status, ...]) -> list[Action]:
        gained = False
        for status in statuses:
            old = self.view.get(status.node_id)
            if old is not None and status.order_key() <= old.order_key():
                continue
            self.view[status.node_id] = status
            gained = gained or gained_capacity(old, status)
        return [Send(self.ctx.node_id, CapacityAvailable())] if gained else []
