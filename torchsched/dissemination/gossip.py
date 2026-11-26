"""
Gossip dissemination: nodes periodically share what they know with a peer.
"""

from __future__ import annotations

from typing import ClassVar

from torchsched.dissemination.capacity import gained_capacity
from torchsched.node import NodeContext
from torchsched.types.actions import Action, Send, SetTimer
from torchsched.types.events import Boot, Event, Received, TimerFired
from torchsched.types.messages import (
    CapacityAvailable,
    LookupReply,
    LookupRequest,
    StatusChanged,
    StatusPush,
)
from torchsched.types.models import NodeId, Status


class GossipDissemination:
    """
    Every `interval` seconds, send this node's status plus every fresh
    record it knows to one random neighbor.

    A node keeps records only about its neighbors, keeps the newest record
    per node, and treats records older than `max_staleness` (by the
    origin's `observed_at`) as unknown. Records therefore spread within a
    neighborhood but do not travel further, so each node's view covers a
    bounded part of the cluster. Lookups return this node's own status
    first, then fresh neighbor records in node id order.
    """

    name: ClassVar[str] = "dissemination"
    accepts: ClassVar[frozenset[type]] = frozenset(
        {StatusChanged, StatusPush, LookupRequest}
    )

    def __init__(self, ctx: NodeContext, *, interval: float, max_staleness: float):
        if interval <= 0:
            raise ValueError("interval must be positive")
        if max_staleness < 0:
            raise ValueError("max_staleness must be nonnegative")
        self.ctx = ctx
        self.interval = interval
        self.max_staleness = max_staleness
        self.view: dict[NodeId, Status] = {}
        self._own: Status | None = None  # last local status, to detect freed GPUs
        self._rng = ctx.rng("gossip-peer")

    def handle(self, event: Event, now: float) -> list[Action]:
        match event:
            case Boot():
                self._own = self.ctx.local_status(now)
                return self._gossip(now)
            case TimerFired(("gossip",)):
                return self._gossip(now)
            case Received(_, StatusChanged(status)):
                gained = gained_capacity(self._own, status)
                self._own = status
                return [self._capacity_available()] if gained else []
            case Received(_, StatusPush(statuses)):
                return self._store(statuses, now)
            case Received(sender, LookupRequest(lookup_id)):
                return [Send(sender, LookupReply(lookup_id, self._candidates(now)))]
        return []

    def _gossip(self, now: float) -> list[Action]:
        actions: list[Action] = [SetTimer(("gossip",), self.interval)]
        peers = sorted(self.ctx.neighbors())
        own = self.ctx.local_status(now)
        if not peers or own is None:
            return actions
        target = self._rng.choice(peers)
        payload = (own, *self._fresh(now))
        actions.append(Send(target, StatusPush(payload)))
        return actions

    def _store(self, statuses: tuple[Status, ...], now: float) -> list[Action]:
        neighbors = self.ctx.neighbors()
        gained = False
        for status in statuses:
            if status.node_id not in neighbors or self._is_stale(status, now):
                continue
            old = self.view.get(status.node_id)
            if old is not None and status.order_key() <= old.order_key():
                continue
            self.view[status.node_id] = status
            gained = gained or gained_capacity(old, status)
        return [self._capacity_available()] if gained else []

    def _candidates(self, now: float) -> tuple[Status, ...]:
        own = self.ctx.local_status(now)
        return (*([own] if own is not None else []), *self._fresh(now))

    def _fresh(self, now: float) -> list[Status]:
        return [
            self.view[node_id]
            for node_id in sorted(self.view)
            if not self._is_stale(self.view[node_id], now)
        ]

    def _is_stale(self, status: Status, now: float) -> bool:
        return now - status.observed_at > self.max_staleness

    def _capacity_available(self) -> Send:
        return Send(self.ctx.node_id, CapacityAvailable())
