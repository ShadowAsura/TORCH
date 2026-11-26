"""
Neighborhood layouts for a fixed list of nodes.
"""

from __future__ import annotations

from collections.abc import Sequence
from random import Random

from torchsched.types.models import NodeId


def ring_with_shortcuts(
    node_ids: Sequence[NodeId], *, radius: int, seed: int
) -> dict[NodeId, frozenset[NodeId]]:
    """
    A small-world layout: a ring plus seeded random shortcuts.

    Each node is linked to the `radius` nodes on either side of it in
    `node_ids` order. Then nodes are shuffled with `seed` and paired off,
    and each pair is linked, so most nodes also get one long-range link.
    Links are symmetric.
    """
    if radius < 1:
        raise ValueError("radius must be positive")
    if len(set(node_ids)) != len(node_ids):
        raise ValueError("node ids must be unique")
    count = len(node_ids)
    offsets = range(1, min(radius, count - 1) + 1)
    links = [
        {(index + sign * offset) % count for offset in offsets for sign in (-1, 1)}
        for index in range(count)
    ]
    shuffled = list(range(count))
    Random(seed).shuffle(shuffled)
    for left, right in zip(shuffled[::2], shuffled[1::2]):
        links[left].add(right)
        links[right].add(left)
    return {
        node_ids[index]: frozenset(node_ids[peer] for peer in peers)
        for index, peers in enumerate(links)
    }


def full_mesh(node_ids: Sequence[NodeId]) -> dict[NodeId, frozenset[NodeId]]:
    """
    Every node is a neighbor of every other node.
    """
    everyone = frozenset(node_ids)
    return {node_id: everyone - {node_id} for node_id in node_ids}
