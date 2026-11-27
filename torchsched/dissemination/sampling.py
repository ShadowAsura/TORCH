"""
Sampling dissemination: probe a few random neighbors when asked, keep no view.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

from torchsched.dissemination.capacity import gained_capacity
from torchsched.node import NodeContext
from torchsched.types.actions import Action, CancelTimer, Send, SetTimer
from torchsched.types.events import Boot, Event, Received, TimerFired
from torchsched.types.messages import (
    CapacityAvailable,
    LookupReply,
    LookupRequest,
    ProbeReply,
    ProbeRequest,
    StatusChanged,
)
from torchsched.types.models import NodeId, Status


@dataclass
class _Lookup:
    requester: NodeId
    waiting: dict[str, NodeId]  # probe request id -> probed node
    replies: dict[NodeId, Status] = field(default_factory=dict)
    order: tuple[NodeId, ...] = ()  # probed nodes, in sampled order


class SamplingDissemination:
    """
    Sparrow-style discovery: no standing view of other nodes.

    For each lookup, probe `sample_size - 1` neighbors chosen at random (all
    of them if there are fewer), and answer with this node's own status
    followed by the probe replies in sampled order. The answer goes out
    when every probe has replied, or after `probe_timeout` seconds with
    whatever has arrived, so a lost probe cannot stall the scheduler.

    Probes from other nodes are answered with this node's worker status. A
    node's own freed GPUs trigger `CapacityAvailable`; capacity freed
    elsewhere is found by the scheduler's retries, since nothing is
    remembered between lookups.
    """

    name: ClassVar[str] = "dissemination"
    accepts: ClassVar[frozenset[type]] = frozenset(
        {LookupRequest, ProbeRequest, ProbeReply, StatusChanged}
    )

    def __init__(self, ctx: NodeContext, *, sample_size: int, probe_timeout: float = 1.0):
        if sample_size < 1:
            raise ValueError("sample_size must be at least 1 (this node itself)")
        if probe_timeout <= 0:
            raise ValueError("probe_timeout must be positive")
        self.ctx = ctx
        self.sample_size = sample_size
        self.probe_timeout = probe_timeout
        self._lookups: dict[str, _Lookup] = {}
        self._probe_owner: dict[str, str] = {}  # probe request id -> lookup id
        self._own: Status | None = None
        self._rng = ctx.rng("sample")
        self._next_id = 0

    def handle(self, event: Event, now: float) -> list[Action]:
        match event:
            case Boot():
                self._own = self.ctx.local_status(now)
                return []
            case Received(sender, LookupRequest(lookup_id)):
                return self._start_lookup(sender, lookup_id, now)
            case Received(sender, ProbeRequest(request_id)):
                status = self.ctx.local_status(now)
                return [Send(sender, ProbeReply(request_id, status))] if status else []
            case Received(_, ProbeReply(request_id, status)):
                return self._probe_replied(request_id, status, now)
            case Received(_, StatusChanged(status)):
                gained = gained_capacity(self._own, status)
                self._own = status
                return [Send(self.ctx.node_id, CapacityAvailable())] if gained else []
            case TimerFired(("probe-timeout", lookup_id)):
                return self._finish(lookup_id, now)
        return []

    def _start_lookup(self, requester: NodeId, lookup_id: str, now: float) -> list[Action]:
        peers = sorted(self.ctx.neighbors())
        sampled = self._rng.sample(peers, min(self.sample_size - 1, len(peers)))
        lookup = _Lookup(requester, waiting={}, order=tuple(sampled))
        self._lookups[lookup_id] = lookup
        if not sampled:
            return self._finish(lookup_id, now)
        actions: list[Action] = [SetTimer(("probe-timeout", lookup_id), self.probe_timeout)]
        for peer in sampled:
            request_id = self._new_id()
            lookup.waiting[request_id] = peer
            self._probe_owner[request_id] = lookup_id
            actions.append(Send(peer, ProbeRequest(request_id)))
        return actions

    def _probe_replied(self, request_id: str, status: Status, now: float) -> list[Action]:
        lookup_id = self._probe_owner.pop(request_id, None)
        if lookup_id is None:
            return []  # late reply for a lookup that already finished
        lookup = self._lookups[lookup_id]
        peer = lookup.waiting.pop(request_id)
        if status.node_id == peer:
            lookup.replies[peer] = status
        if lookup.waiting:
            return []
        return [CancelTimer(("probe-timeout", lookup_id)), *self._finish(lookup_id, now)]

    def _finish(self, lookup_id: str, now: float) -> list[Action]:
        lookup = self._lookups.pop(lookup_id, None)
        if lookup is None:
            return []
        for request_id in lookup.waiting:
            self._probe_owner.pop(request_id, None)
        own = self.ctx.local_status(now)
        candidates = ([own] if own is not None else []) + [
            lookup.replies[peer] for peer in lookup.order if peer in lookup.replies
        ]
        return [Send(lookup.requester, LookupReply(lookup_id, tuple(candidates)))]

    def _new_id(self) -> str:
        self._next_id += 1
        return f"{self.ctx.node_id}:{self.ctx.incarnation}:probe-{self._next_id}"
