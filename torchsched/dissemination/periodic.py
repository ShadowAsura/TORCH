"""
Periodic dissemination: every node pushes its own status to all neighbors.
"""

from __future__ import annotations

from typing import ClassVar

from torchsched.dissemination.gossip import GossipDissemination
from torchsched.types.actions import Action, Send
from torchsched.types.messages import StatusPush
from torchsched.types.models import NodeId, Status


class PeriodicDissemination(GossipDissemination):
    """
    Navigator-style bounded staleness: every `interval` seconds, send this
    node's own status, and nothing else, to every neighbor.

    Records are never relayed, so each record in a view came straight from
    its node and is at most one interval plus one network delay old. Records
    older than `max_staleness` are treated as unknown. Storing records,
    lookups, and capacity notices work exactly as in `GossipDissemination`.
    """

    timer: ClassVar[tuple[str]] = ("periodic",)

    def _push(self, own: Status, peers: list[NodeId], now: float) -> list[Action]:
        push = StatusPush((own,))
        return [Send(peer, push) for peer in peers]
