"""
Builders that assemble nodes into a cluster for one scheduling design.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from torchsched.dissemination.centralized import CentralizedDissemination
from torchsched.dissemination.gossip import GossipDissemination
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


def gossip(
    workers: Sequence[NodeSpec],
    *,
    neighborhoods: Mapping[NodeId, frozenset[NodeId]],
    placement: PlacementPolicy,
    interval: float,
    max_staleness: float,
    retry_interval: float,
    seed: int = 0,
    lease_seconds: float = 30.0,
) -> list[Node]:
    """
    Every node runs a worker and a scheduler and gossips with its neighbors.

    `neighborhoods` maps each node to its neighbors, for example from
    `topology.ring_with_shortcuts` or `topology.full_mesh`. Jobs may be
    submitted to any node, which places them using its own view.

    TODO: forward jobs to other schedulers. Until then, a job waits at the
    node it was submitted to, so a job that no GPU in that node's
    neighborhood can run waits forever.
    """
    missing = {spec.node_id for spec in workers} - set(neighborhoods)
    if missing:
        raise ValueError(f"no neighborhood for {sorted(missing)}")
    nodes = []
    for spec in workers:
        ctx = NodeContext(spec, incarnation=1, seed=seed)
        nodes.append(
            Node(
                ctx,
                worker=Worker(ctx, lease_seconds=lease_seconds),
                membership=StaticMembership(neighborhoods[spec.node_id]),
                dissemination=GossipDissemination(
                    ctx, interval=interval, max_staleness=max_staleness
                ),
                scheduler=Scheduler(
                    ctx, placement=placement, retry_interval=retry_interval
                ),
            )
        )
    return nodes
