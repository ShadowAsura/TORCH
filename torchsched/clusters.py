"""
Builders that assemble nodes into a cluster for one scheduling design.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from functools import partial
from typing import Any

from torchsched import topology
from torchsched.codec import from_data
from torchsched.dissemination.centralized import CentralizedDissemination
from torchsched.dissemination.gossip import GossipDissemination
from torchsched.dissemination.periodic import PeriodicDissemination
from torchsched.dissemination.sampling import SamplingDissemination
from torchsched.forwarding.ring import RingForwarding
from torchsched.interfaces.component import DisseminationRole
from torchsched.interfaces.policies import ForwardingPolicy, PlacementPolicy
from torchsched.membership.static import StaticMembership
from torchsched.node import Node, NodeContext
from torchsched.placement.first_fit import FirstFit
from torchsched.scheduler import Scheduler
from torchsched.types.models import GpuGroup, NodeId, NodeSpec
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

    def worker(spec: NodeSpec, incarnation: int) -> Node:
        ctx = NodeContext(spec, incarnation=incarnation, seed=seed)
        return Node(
            ctx,
            worker=Worker(ctx, lease_seconds=lease_seconds),
            membership=StaticMembership(),
            dissemination=CentralizedDissemination(ctx, coordinator=coordinator),
        )

    def coordinator_node(incarnation: int) -> Node:
        ctx = NodeContext(NodeSpec(coordinator), incarnation=incarnation, seed=seed)
        return Node(
            ctx,
            membership=StaticMembership(),
            dissemination=CentralizedDissemination(ctx, coordinator=coordinator),
            scheduler=Scheduler(ctx, placement=placement),
        )

    nodes = [_restartable(partial(worker, spec)) for spec in workers]
    return [*nodes, _restartable(coordinator_node)]


def gossip(
    workers: Sequence[NodeSpec],
    *,
    neighborhoods: Mapping[NodeId, frozenset[NodeId]],
    placement: PlacementPolicy,
    forwarding: ForwardingPolicy | None,
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

    A job that no node in a scheduler's view has room for is handed to the
    next scheduler chosen by `forwarding`, for example
    `forwarding.ring.RingForwarding`. With `forwarding=None`, a job stays
    at the node it was submitted to, so a job that nothing in that node's
    neighborhood can run waits forever.
    """
    return _peer_to_peer(
        workers,
        neighborhoods=neighborhoods,
        placement=placement,
        forwarding=forwarding,
        retry_interval=retry_interval,
        seed=seed,
        lease_seconds=lease_seconds,
        dissemination=lambda ctx: GossipDissemination(
            ctx, interval=interval, max_staleness=max_staleness
        ),
    )


def periodic(
    workers: Sequence[NodeSpec],
    *,
    neighborhoods: Mapping[NodeId, frozenset[NodeId]],
    placement: PlacementPolicy,
    forwarding: ForwardingPolicy | None,
    interval: float,
    max_staleness: float,
    retry_interval: float,
    seed: int = 0,
    lease_seconds: float = 30.0,
) -> list[Node]:
    """
    Like `gossip`, but every `interval` seconds each node sends only its own
    status to every neighbor (Navigator-style bounded staleness).
    """
    return _peer_to_peer(
        workers,
        neighborhoods=neighborhoods,
        placement=placement,
        forwarding=forwarding,
        retry_interval=retry_interval,
        seed=seed,
        lease_seconds=lease_seconds,
        dissemination=lambda ctx: PeriodicDissemination(
            ctx, interval=interval, max_staleness=max_staleness
        ),
    )


def sampling(
    workers: Sequence[NodeSpec],
    *,
    neighborhoods: Mapping[NodeId, frozenset[NodeId]],
    placement: PlacementPolicy,
    forwarding: ForwardingPolicy | None,
    sample_size: int,
    retry_interval: float,
    probe_timeout: float = 1.0,
    seed: int = 0,
    lease_seconds: float = 30.0,
) -> list[Node]:
    """
    Like `gossip`, but schedulers probe `sample_size - 1` random neighbors
    per placement round instead of keeping a view (Sparrow-style).
    """
    return _peer_to_peer(
        workers,
        neighborhoods=neighborhoods,
        placement=placement,
        forwarding=forwarding,
        retry_interval=retry_interval,
        seed=seed,
        lease_seconds=lease_seconds,
        dissemination=lambda ctx: SamplingDissemination(
            ctx, sample_size=sample_size, probe_timeout=probe_timeout
        ),
    )


def _peer_to_peer(
    workers: Sequence[NodeSpec],
    *,
    neighborhoods: Mapping[NodeId, frozenset[NodeId]],
    placement: PlacementPolicy,
    forwarding: ForwardingPolicy | None,
    retry_interval: float,
    seed: int,
    lease_seconds: float,
    dissemination: Callable[[NodeContext], DisseminationRole],
) -> list[Node]:
    """
    Every node runs a worker, a scheduler, and the given dissemination.
    """
    missing = {spec.node_id for spec in workers} - set(neighborhoods)
    if missing:
        raise ValueError(f"no neighborhood for {sorted(missing)}")

    def node(spec: NodeSpec, incarnation: int) -> Node:
        ctx = NodeContext(spec, incarnation=incarnation, seed=seed)
        return Node(
            ctx,
            worker=Worker(ctx, lease_seconds=lease_seconds),
            membership=StaticMembership(neighborhoods[spec.node_id]),
            dissemination=dissemination(ctx),
            scheduler=Scheduler(
                ctx,
                placement=placement,
                forwarding=forwarding,
                retry_interval=retry_interval,
            ),
        )

    return [_restartable(partial(node, spec)) for spec in workers]


def _restartable(make: Callable[[int], Node]) -> Node:
    """
    The first incarnation of a node. Every incarnation can rebuild itself
    after a crash.
    """

    def build(incarnation: int) -> Node:
        node = make(incarnation)
        node.rebuild = build
        return node

    return build(1)


def from_config(config: Mapping[str, Any]) -> list[Node]:
    """
    Build every node of a cluster from a JSON-style config.

    ```
    {
      "design": "centralized" | "gossip" | "periodic" | "sampling",
      "seed": 0,                              # optional
      "lease_seconds": 30,                    # optional
      "placement": {"backfill": false},       # optional
      "workers": [
        {"id": "w1", "gpus": [{"gpu_type": "A100", "count": 2, "vram_gb": 80}]}
      ],
      "centralized": {"coordinator": "coordinator"},   # optional
      "gossip": {
        "interval": 0.5, "max_staleness": 3.0, "retry_interval": 0.25,
        "topology": "ring" | "full",   # optional, default "ring"
        "radius": 1,                   # optional, ring only
        "forwarding": "ring" | null    # optional, default "ring"
      },
      "periodic": {same keys as gossip},
      "sampling": {
        "sample_size": 2, "retry_interval": 0.25,
        "probe_timeout": 1.0,          # optional
        "topology", "radius", "forwarding": as for gossip
      }
    }
    ```

    The ring order is the order of `workers`. Other keys (for example, a
    runtime's addresses) are ignored, so one file can configure every
    process of a deployment.
    """
    design = config.get("design")
    seed = int(config.get("seed", 0))
    lease_seconds = float(config.get("lease_seconds", 30.0))
    placement = FirstFit(backfill=bool(config.get("placement", {}).get("backfill", False)))
    workers = [
        NodeSpec(
            str(worker["id"]),
            [from_data(GpuGroup, group) for group in worker.get("gpus", [])],
        )
        for worker in config.get("workers", [])
    ]
    if not workers:
        raise ValueError("config needs at least one worker")

    if design == "centralized":
        coordinator = config.get("centralized", {}).get("coordinator", "coordinator")
        return centralized(
            workers,
            placement=placement,
            coordinator=coordinator,
            seed=seed,
            lease_seconds=lease_seconds,
        )
    if design in ("gossip", "periodic", "sampling"):
        options = config.get(design, {})
        neighborhoods, forwarding = _peer_layout(workers, options, seed)
        shared = dict(
            neighborhoods=neighborhoods,
            placement=placement,
            forwarding=forwarding,
            retry_interval=float(options["retry_interval"]),
            seed=seed,
            lease_seconds=lease_seconds,
        )
        if design in ("gossip", "periodic"):
            build = gossip if design == "gossip" else periodic
            return build(
                workers,
                interval=float(options["interval"]),
                max_staleness=float(options["max_staleness"]),
                **shared,
            )
        return sampling(
            workers,
            sample_size=int(options["sample_size"]),
            probe_timeout=float(options.get("probe_timeout", 1.0)),
            **shared,
        )
    raise ValueError(f"unknown design {design!r}")


def _peer_layout(
    workers: Sequence[NodeSpec], options: Mapping[str, Any], seed: int
) -> tuple[dict[NodeId, frozenset[NodeId]], ForwardingPolicy | None]:
    """
    Neighborhoods and forwarding from a peer-to-peer design's config section.
    """
    ids = [spec.node_id for spec in workers]
    layout = options.get("topology", "ring")
    if layout == "ring":
        neighborhoods = topology.ring_with_shortcuts(
            ids, radius=int(options.get("radius", 1)), seed=seed
        )
    elif layout == "full":
        neighborhoods = topology.full_mesh(ids)
    else:
        raise ValueError(f"unknown topology {layout!r}")
    forwarding_name = options.get("forwarding", "ring")
    if forwarding_name not in ("ring", None):
        raise ValueError(f"unknown forwarding policy {forwarding_name!r}")
    return neighborhoods, RingForwarding(ids) if forwarding_name == "ring" else None
