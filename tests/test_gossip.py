"""
Tests for neighborhood layouts, gossip dissemination, and gossip clusters.
"""

from collections import defaultdict

import pytest

import distributed as legacy
from torchsched import clusters, topology
from torchsched.dissemination.gossip import GossipDissemination
from torchsched.membership.static import StaticMembership
from torchsched.node import Node, NodeContext
from torchsched.placement.first_fit import FirstFit
from torchsched.runtime.sim import Simulator
from torchsched.types.actions import Send, SetTimer
from torchsched.types.events import Boot, Received, TimerFired
from torchsched.types.messages import (
    CapacityAvailable,
    LookupReply,
    LookupRequest,
    StatusChanged,
    StatusPush,
)
from torchsched.types.metrics import JobCompleted
from torchsched.types.models import GpuGroup, Job, NodeSpec, Status
from torchsched.worker import Worker

# ==================================================
# Topology
# ==================================================


@pytest.mark.parametrize("count,radius,seed", [(12, 1, 42), (7, 2, 3), (2, 1, 0), (1, 1, 5)])
def test_ring_with_shortcuts_matches_the_legacy_layout(count, radius, seed):
    ids = [f"n{i}" for i in range(count)]
    layout = topology.ring_with_shortcuts(ids, radius=radius, seed=seed)
    expected = legacy.neighborhoods(count, radius, seed)
    assert layout == {
        ids[i]: frozenset(ids[peer] for peer in peers) for i, peers in enumerate(expected)
    }


def test_ring_with_shortcuts_is_symmetric_and_includes_the_ring():
    ids = [f"n{i}" for i in range(10)]
    layout = topology.ring_with_shortcuts(ids, radius=2, seed=1)
    for i, node in enumerate(ids):
        assert node not in layout[node]
        for offset in (1, 2):
            assert ids[(i + offset) % 10] in layout[node]
            assert ids[(i - offset) % 10] in layout[node]
        for peer in layout[node]:
            assert node in layout[peer]
    with pytest.raises(ValueError):
        topology.ring_with_shortcuts(ids, radius=0, seed=1)
    with pytest.raises(ValueError):
        topology.ring_with_shortcuts(["a", "a"], radius=1, seed=1)


def test_full_mesh():
    assert topology.full_mesh(["a", "b", "c"]) == {
        "a": frozenset({"b", "c"}),
        "b": frozenset({"a", "c"}),
        "c": frozenset({"a", "b"}),
    }


# ==================================================
# Gossip component
# ==================================================

GPU = GpuGroup("A100", 2, 80)


def spec(node_id):
    return NodeSpec(node_id, [GPU])


def record(node_id, version, free=2, at=0.0, incarnation=1):
    return Status(spec(node_id), (free,), incarnation, version, at)


def gossip_node(node_id="a", neighbors=("b", "c"), *, staleness=3.0, seed=0):
    """
    A node with a real worker and gossip, but no scheduler.
    """
    ctx = NodeContext(spec(node_id), incarnation=1, seed=seed)
    dissemination = GossipDissemination(ctx, interval=0.5, max_staleness=staleness)
    node = Node(
        ctx,
        worker=Worker(ctx),
        membership=StaticMembership(neighbors),
        dissemination=dissemination,
    )
    return node, dissemination


def pushes(actions):
    return [a for a in actions if isinstance(a, Send) and isinstance(a.msg, StatusPush)]


def capacity_notices(actions):
    return [a for a in actions if isinstance(a, Send) and isinstance(a.msg, CapacityAvailable)]


def test_each_round_pushes_own_status_to_one_neighbor_and_rearms():
    node, _ = gossip_node()
    actions = node.handle(Boot(), 0.0)
    assert SetTimer(("dissemination", "gossip"), 0.5) in actions
    [push] = pushes(actions)
    assert push.to in {"b", "c"}
    assert [s.node_id for s in push.msg.statuses] == ["a"]

    targets = {
        pushes(node.handle(TimerFired(("dissemination", "gossip")), t / 2))[0].to
        for t in range(1, 20)
    }
    assert targets == {"b", "c"}


def test_peer_choice_is_deterministic_per_seed():
    def targets(seed):
        node, _ = gossip_node(neighbors=("b", "c", "d", "e"), seed=seed)
        node.handle(Boot(), 0.0)
        return [
            pushes(node.handle(TimerFired(("dissemination", "gossip")), float(t)))[0].to
            for t in range(10)
        ]

    assert targets(1) == targets(1)
    assert targets(1) != targets(2)


def test_push_carries_fresh_records_only():
    node, view = gossip_node()
    node.handle(Boot(), 0.0)
    node.handle(Received("b", StatusPush((record("b", 1, at=0.0), record("c", 1, at=2.0)))), 2.0)
    [push] = pushes(node.handle(TimerFired(("dissemination", "gossip")), 4.0))
    # b's record is 4 s old (> 3 s staleness); c's is 2 s old.
    assert [s.node_id for s in push.msg.statuses] == ["a", "c"]


def test_received_records_are_filtered_and_ordered():
    node, view = gossip_node()
    node.handle(Boot(), 0.0)
    node.handle(
        Received(
            "b",
            StatusPush(
                (
                    record("b", 5),
                    record("z", 1),  # not a neighbor
                    record("a", 9),  # this node itself
                    record("c", 1, at=-10.0),  # already stale
                )
            ),
        ),
        0.0,
    )
    assert set(view.view) == {"b"}
    # An older record for b never replaces a newer one; a restart does.
    node.handle(Received("c", StatusPush((record("b", 3, free=0),))), 0.0)
    assert view.view["b"].version == 5
    node.handle(Received("c", StatusPush((record("b", 0, incarnation=2),))), 0.0)
    assert view.view["b"].incarnation == 2


def test_capacity_notices_only_when_capacity_appears():
    node, _ = gossip_node()
    node.handle(Boot(), 0.0)
    first = node.handle(Received("b", StatusPush((record("b", 1, free=0),))), 0.0)
    assert capacity_notices(first)  # a new node counts as new capacity
    busier = node.handle(Received("b", StatusPush((record("b", 2, free=0),))), 0.0)
    assert not capacity_notices(busier)
    freed = node.handle(Received("b", StatusPush((record("b", 3, free=2),))), 0.0)
    assert capacity_notices(freed)

    reserved = Status(spec("a"), (0,), 1, 1, 1.0)
    assert not capacity_notices(node.handle(Received("a", StatusChanged(reserved)), 1.0))
    released = Status(spec("a"), (2,), 1, 2, 2.0)
    assert capacity_notices(node.handle(Received("a", StatusChanged(released)), 2.0))


def test_lookup_returns_own_status_then_fresh_neighbors():
    node, _ = gossip_node()
    node.handle(Boot(), 0.0)
    node.handle(Received("c", StatusPush((record("c", 1, at=1.0), record("b", 1, at=0.5)))), 1.0)
    [reply] = node.handle(Received("a", LookupRequest("l1")), 2.0)
    assert isinstance(reply.msg, LookupReply)
    assert [s.node_id for s in reply.msg.candidates] == ["a", "b", "c"]
    [later] = node.handle(Received("a", LookupRequest("l2")), 3.75)
    assert [s.node_id for s in later.msg.candidates] == ["a", "c"]  # b went stale


def test_node_without_neighbors_only_rearms():
    node, _ = gossip_node(neighbors=())
    assert node.handle(Boot(), 0.0) == [SetTimer(("dissemination", "gossip"), 0.5)]


# ==================================================
# End to end
# ==================================================

MIXED_CLUSTER = [NodeSpec(f"l40-{i}", [GpuGroup("L40", 2, 48)]) for i in range(1, 5)] + [
    NodeSpec(f"a100-{i}", [GpuGroup("A100", 2, 80)]) for i in range(1, 5)
]


def workload(count=60, seed=9):
    """
    Jobs every node can run, so placement never depends on forwarding.
    """
    import random

    rng = random.Random(seed)
    at, jobs = 0.0, []
    for i in range(count):
        at += rng.expovariate(3.0)
        runtime = rng.uniform(1, 5)
        jobs.append((Job(f"j{i}", rng.choice([1, 2]), 24, {"L40": runtime, "A100": runtime * 0.75}), at))
    return jobs


def run(neighborhoods, jobs, *, latency=0.02, seed=1, backfill=False):
    ids = [s.node_id for s in MIXED_CLUSTER]
    sim = Simulator(
        clusters.gossip(
            MIXED_CLUSTER,
            neighborhoods=neighborhoods,
            placement=FirstFit(backfill=backfill),
            forwarding=None,
            interval=0.5,
            max_staleness=3.0,
            retry_interval=0.25,
            seed=seed,
        ),
        latency=latency,
    )
    for i, (job, at) in enumerate(jobs):
        sim.submit(job, at=at, node_id=ids[i % len(ids)])
    return sim.run(until=1_000, until_jobs_done=True)


def assert_never_overbooked(result):
    """
    At every instant, each worker runs at most as many GPUs as it has.

    Uses the simulator's execution log, not scheduler metrics: a scheduler
    records a start or finish when the message reaches it, which can be
    later than when the worker actually started or finished the job.
    """
    changes = defaultdict(list)
    for run in result.executions:
        changes[run.node_id] += [(run.start, run.gpu_count), (run.end, -run.gpu_count)]
    for worker, deltas in changes.items():
        in_use = 0
        # At a tie, the release (negative) sorts before the start.
        for _, delta in sorted(deltas):
            in_use += delta
            assert 0 <= in_use <= 2, worker


@pytest.mark.parametrize("layout", ["ring", "full"])
def test_gossip_cluster_completes_every_job_without_overbooking(layout):
    ids = [s.node_id for s in MIXED_CLUSTER]
    neighborhoods = (
        topology.ring_with_shortcuts(ids, radius=1, seed=4)
        if layout == "ring"
        else topology.full_mesh(ids)
    )
    jobs = workload()
    result = run(neighborhoods, jobs)
    completed = {o.metric.job_id for o in result.observations if isinstance(o.metric, JobCompleted)}
    assert completed == {job.job_id for job, _ in jobs}
    assert result.end_time < 1_000  # stopped when the last job finished
    assert result.messages["StatusPush"] > 0
    assert_never_overbooked(result)


def test_gossip_runs_are_deterministic():
    ids = [s.node_id for s in MIXED_CLUSTER]
    jobs = workload()
    first = run(topology.full_mesh(ids), jobs, seed=3)
    second = run(topology.full_mesh(ids), jobs, seed=3)
    assert first.observations == second.observations
    assert first.messages == second.messages


def test_simulator_stops_when_jobs_finish_despite_periodic_timers():
    ids = [s.node_id for s in MIXED_CLUSTER]
    job = Job("only", 1, 24, {"L40": 2.0, "A100": 2.0})
    result = run(topology.full_mesh(ids), [(job, 1.0)])
    assert result.end_time == 3.0  # placed from the local worker at t=1
