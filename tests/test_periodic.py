"""
Tests for periodic dissemination and periodic clusters.
"""

from collections import defaultdict

from torchsched import clusters, topology
from torchsched.dissemination.periodic import PeriodicDissemination
from torchsched.forwarding.ring import RingForwarding
from torchsched.membership.static import StaticMembership
from torchsched.node import Node, NodeContext
from torchsched.placement.first_fit import FirstFit
from torchsched.runtime.sim import Simulator
from torchsched.types.actions import Send, SetTimer
from torchsched.types.events import Boot, Received, TimerFired
from torchsched.types.messages import LookupRequest, StatusPush
from torchsched.types.metrics import JobCompleted
from torchsched.types.models import GpuGroup, NodeSpec, Status
from torchsched.worker import Worker
from torchsched.workloads import poisson_workload

# ==================================================
# Component
# ==================================================

TIMER = ("dissemination", "periodic")


def spec(node_id):
    return NodeSpec(node_id, [GpuGroup("A100", 2, 80)])


def record(node_id, version=1, at=0.0):
    return Status(spec(node_id), (2,), 1, version, at)


def periodic_node(neighbors=("b", "c", "d"), *, worker=True):
    ctx = NodeContext(spec("a") if worker else NodeSpec("a"), incarnation=1, seed=0)
    return Node(
        ctx,
        worker=Worker(ctx) if worker else None,
        membership=StaticMembership(neighbors),
        dissemination=PeriodicDissemination(ctx, interval=0.5, max_staleness=3.0),
    )


def pushes(actions):
    return [a for a in actions if isinstance(a, Send) and isinstance(a.msg, StatusPush)]


def test_each_interval_pushes_only_own_status_to_every_neighbor():
    node = periodic_node()
    actions = node.handle(Boot(), 0.0)
    assert SetTimer(TIMER, 0.5) in actions
    sent = pushes(actions)
    assert sorted(p.to for p in sent) == ["b", "c", "d"]
    assert all(p.msg.statuses == (node.worker.status(0.0),) for p in sent)

    # Records learned from neighbors are never relayed.
    node.handle(Received("b", StatusPush((record("b", at=0.2),))), 0.2)
    sent = pushes(node.handle(TimerFired(TIMER), 0.5))
    assert len(sent) == 3
    assert all([s.node_id for s in p.msg.statuses] == ["a"] for p in sent)


def test_received_records_are_used_until_stale():
    node = periodic_node()
    node.handle(Boot(), 0.0)
    node.handle(Received("b", StatusPush((record("b", at=1.0),))), 1.0)
    node.handle(Received("c", StatusPush((record("c", at=0.5),))), 1.0)
    [reply] = node.handle(Received("a", LookupRequest("l1")), 2.0)
    assert [s.node_id for s in reply.msg.candidates] == ["a", "b", "c"]
    [later] = node.handle(Received("a", LookupRequest("l2")), 3.75)
    assert [s.node_id for s in later.msg.candidates] == ["a", "b"]  # c went stale


def test_nodes_without_a_worker_or_neighbors_only_rearm():
    assert periodic_node(worker=False).handle(Boot(), 0.0) == [SetTimer(TIMER, 0.5)]
    assert periodic_node(neighbors=()).handle(Boot(), 0.0) == [SetTimer(TIMER, 0.5)]


# ==================================================
# End to end
# ==================================================

CLUSTER = [NodeSpec(f"l40-{i}", [GpuGroup("L40", 2, 48)]) for i in range(1, 7)] + [
    NodeSpec(f"a100-{i}", [GpuGroup("A100", 2, 80)]) for i in range(1, 4)
]
IDS = [s.node_id for s in CLUSTER]


def run(seed=4):
    neighborhoods = topology.ring_with_shortcuts(IDS, radius=1, seed=seed)
    sim = Simulator(
        clusters.periodic(
            CLUSTER,
            neighborhoods=neighborhoods,
            placement=FirstFit(backfill=True),
            forwarding=RingForwarding(IDS),
            interval=0.5,
            max_staleness=3.0,
            retry_interval=0.25,
            seed=seed,
        ),
        latency=0.02,
    )
    # The default workload includes 64 GB jobs only A100 nodes can run.
    workload = poisson_workload(seed, 80, 4.0)
    for i, submission in enumerate(workload):
        sim.submit(submission.job, at=submission.at, node_id=IDS[i % len(IDS)])
    return neighborhoods, workload, sim.run(until=2_000, until_jobs_done=True)


def test_periodic_cluster_completes_every_job_without_overbooking():
    neighborhoods, workload, result = run()
    done = {o.metric.job_id for o in result.observations if isinstance(o.metric, JobCompleted)}
    assert done == {s.job.job_id for s in workload}

    # Each node pushes to every neighbor once per interval (plus at boot).
    rounds = int(result.end_time / 0.5) + 1
    degree = sum(len(peers) for peers in neighborhoods.values())
    assert 0 < result.messages["StatusPush"] - degree * (rounds - 1) <= degree

    usage = defaultdict(list)
    for execution in result.executions:
        usage[execution.node_id] += [(execution.start, execution.gpu_count), (execution.end, -execution.gpu_count)]
    for node_id, deltas in usage.items():
        in_use = 0
        for _, delta in sorted(deltas):
            in_use += delta
            assert 0 <= in_use <= 2, node_id


def test_periodic_runs_are_deterministic():
    _, _, first = run(seed=7)
    _, _, second = run(seed=7)
    assert first.observations == second.observations
    assert first.messages == second.messages


def test_from_config_builds_periodic_nodes():
    nodes = clusters.from_config(
        {
            "design": "periodic",
            "workers": [{"id": i, "gpus": [{"gpu_type": "A100", "count": 1, "vram_gb": 80}]} for i in "abc"],
            "periodic": {"interval": 0.5, "max_staleness": 3.0, "retry_interval": 0.25, "topology": "full"},
        }
    )
    assert [type(n.dissemination) for n in nodes] == [PeriodicDissemination] * 3
    assert all(n.scheduler is not None and n.worker is not None for n in nodes)
    assert nodes[0].ctx.neighbors() == frozenset("bc")
