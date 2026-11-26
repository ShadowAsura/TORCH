"""
Pure decision functions that components call.

Policies keep no state between calls, do no I/O and read no clock, so each
one can be unit-tested on its own. Policies that need randomness take a
seeded `random.Random` from NodeContext when they are constructed.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from torchsched.types.models import Assignment, Job, NodeId, Status


class PlacementPolicy(Protocol):
    """
    Decide which pending jobs to try on which GPU groups.

    This works on a batch of jobs rather than one job at a time, because
    some policies (Sia, backfilling) trade jobs off against each other. FIFO
    first-fit is the special case that walks `pending` in order.

    Each Assignment names a node and one of its GPU groups. Assignments are
    tried in the order returned. A plan must not commit more than a group's
    free GPUs (`Status.free_gpus[group]`). It may return fewer assignments
    than there are jobs; unassigned jobs stay queued.
    """

    name: str

    def plan(
        self, pending: Sequence[Job], candidates: Sequence[Status], now: float
    ) -> list[Assignment]: ...


class ForwardingPolicy(Protocol):
    """
    Choose the next scheduler for a job this node cannot place.

    `visited` lists the schedulers that tried the job on its current lap,
    in order, ending with this node. Returning `None` ends the lap: the job
    stays here and starts a new lap after the retry timer.
    """

    name: str

    def next_hop(
        self,
        job: Job,
        visited: Sequence[NodeId],
        neighbors: frozenset[NodeId],
    ) -> NodeId | None: ...


class Topology(Protocol):
    """
    Map the current member set to this node's neighborhood (small world).
    """

    name: str

    def neighbors(
        self, node_id: NodeId, members: frozenset[NodeId]
    ) -> frozenset[NodeId]: ...
