"""
Forward jobs to the next scheduler around a ring.
"""

from __future__ import annotations

from collections.abc import Sequence

from torchsched.types.models import Job, NodeId


class RingForwarding:
    """
    Send a job to the node after the current one in `order`, wrapping around.

    A lap ends when the next node has already tried the job on this lap, so
    a job visits each scheduler at most once per lap. The next node must be
    a neighbor; with `topology.ring_with_shortcuts` built from the same
    `order`, it always is.
    """

    name = "ring"

    def __init__(self, order: Sequence[NodeId]):
        if len(set(order)) != len(order):
            raise ValueError("ring order must not repeat nodes")
        self._successor = {
            node: order[(index + 1) % len(order)] for index, node in enumerate(order)
        }

    def next_hop(
        self,
        job: Job,
        visited: Sequence[NodeId],
        neighbors: frozenset[NodeId],
    ) -> NodeId | None:
        successor = self._successor.get(visited[-1])
        if successor is None or successor in visited or successor not in neighbors:
            return None
        return successor
