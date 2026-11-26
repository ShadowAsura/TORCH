"""
Builders that assemble nodes into a cluster for one scheduling design.
"""

from __future__ import annotations

from collections.abc import Sequence

from torchsched.dissemination.centralized import CentralizedDissemination
from torchsched.interfaces.policies import PlacementPolicy
from torchsched.membership.static import StaticMembership
from torchsched.node import Node, NodeContext
from torchsched.scheduler import Scheduler
from torchsched.types.models import NodeId, NodeSpec
from torchsched.worker import Worker


def centralized(
    workers: Sequence[NodeSpec],
    *,
    placement: PlacementPolicy,
    coordinator: NodeId = "coordinator",
    seed: int = 0,
    lease_seconds: float = 30.0,
) -> list[Node]:
    """
    One coordinator without GPUs schedules every job; workers report to it.

    Submit all jobs to `coordinator`. The coordinator is last in the list,
    so with a deterministic runtime the workers boot (and report) first.
    """
    if any(spec.node_id == coordinator for spec in workers):
        raise ValueError(f"worker id {coordinator!r} collides with the coordinator")
    nodes = []
    for spec in workers:
        ctx = NodeContext(spec, incarnation=1, seed=seed)
        nodes.append(
            Node(
                ctx,
                worker=Worker(ctx, lease_seconds=lease_seconds),
                membership=StaticMembership(),
                dissemination=CentralizedDissemination(ctx, coordinator=coordinator),
            )
        )
    ctx = NodeContext(NodeSpec(coordinator), incarnation=1, seed=seed)
    nodes.append(
        Node(
            ctx,
            membership=StaticMembership(),
            dissemination=CentralizedDissemination(ctx, coordinator=coordinator),
            scheduler=Scheduler(ctx, placement=placement),
        )
    )
    return nodes
