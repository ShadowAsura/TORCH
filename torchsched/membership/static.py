"""
Membership from a fixed list.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import ClassVar

from torchsched.types.actions import Action
from torchsched.types.events import Event
from torchsched.types.models import NodeId


class StaticMembership:
    """
    A fixed neighborhood that never changes.
    """

    name: ClassVar[str] = "membership"
    accepts: ClassVar[frozenset[type]] = frozenset()

    def __init__(self, neighbors: Iterable[NodeId] = ()):
        self._neighbors = frozenset(neighbors)

    def handle(self, event: Event, now: float) -> list[Action]:
        return []

    def neighbors(self) -> frozenset[NodeId]:
        return self._neighbors
