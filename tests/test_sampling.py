"""
Tests for sampling dissemination and sampling clusters.
"""

from collections import defaultdict

from torchsched import clusters, topology
from torchsched.dissemination.sampling import SamplingDissemination
from torchsched.forwarding.ring import RingForwarding
from torchsched.membership.static import StaticMembership
from torchsched.node import Node, NodeContext
from torchsched.placement.first_fit import FirstFit
from torchsched.runtime.sim import Simulator
from torchsched.types.actions import CancelTimer, Send, SetTimer
from torchsched.types.events import Boot, Received, TimerFired
from torchsched.types.messages import (
    CapacityAvailable,
    LookupReply,
    LookupRequest,
    ProbeReply,
    ProbeRequest,
    StatusChanged,
)
from torchsched.types.metrics import JobCompleted
from torchsched.types.models import GpuGroup, NodeSpec, Status
from torchsched.worker import Worker
from torchsched.workloads import poisson_workload

# ==================================================
# Component
# ==================================================


def spec(node_id):
    return NodeSpec(node_id, [GpuGroup("A100", 2, 80)])


def record(node_id, free=2):
    return Status(spec(node_id), (free,), 1, 1, 0.0)


def sampling_node(neighbors=("b", "c", "d"), *, sample_size=3, seed=0):
    ctx = NodeContext(spec("a"), incarnation=1, seed=seed)
    node = Node(
        ctx,
        worker=Worker(ctx),
        membership=StaticMembership(neighbors),
        dissemination=SamplingDissemination(ctx, sample_size=sample_size, probe_timeout=0.5),
    )
    node.handle(Boot(), 0.0)
    return node


def probes(actions):
    return [a for a in actions if isinstance(a, Send) and isinstance(a.msg, ProbeRequest)]


def replies(actions):
    return [a.msg for a in actions if isinstance(a, Send) and isinstance(a.msg, LookupReply)]


def test_lookup_probes_a_sample_and_answers_when_all_reply():
    node = sampling_node(sample_size=3)
    actions = node.handle(Received("a", LookupRequest("l1")), 0.0)
    sent = probes(actions)
    assert len(sent) == 2  # sample_size counts this node itself
    assert len({p.to for p in sent}) == 2 and {p.to for p in sent} <= {"b", "c", "d"}
    assert SetTimer(("dissemination", "probe-timeout", "l1"), 0.5) in actions
    assert not replies(actions)

    first, second = sent
    assert node.handle(Received(first.to, ProbeReply(first.msg.request_id, record(first.to))), 0.1) == []
    actions = node.handle(Received(second.to, ProbeReply(second.msg.request_id, record(second.to, 0))), 0.1)
    [reply] = replies(actions)
    assert CancelTimer(("dissemination", "probe-timeout", "l1")) in actions
    # Own status first, then the probed nodes in sampled order.
    assert [s.node_id for s in reply.candidates] == ["a", first.to, second.to]
    assert reply.candidates[2].free_gpus == (0,)


def test_sample_is_seeded_and_varies_between_lookups():
    def sampled(seed):
        node = sampling_node(neighbors=tuple("bcdefg"), sample_size=2, seed=seed)
        return [probes(node.handle(Received("a", LookupRequest(f"l{i}")), 0.0))[0].to for i in range(12)]

    assert sampled(1) == sampled(1)
    assert len(set(sampled(1))) > 1


def test_timeout_answers_with_the_replies_that_arrived():
    node = sampling_node(sample_size=3)
    first, second = probes(node.handle(Received("a", LookupRequest("l1")), 0.0))
    node.handle(Received(first.to, ProbeReply(first.msg.request_id, record(first.to))), 0.1)
    [reply] = replies(node.handle(TimerFired(("dissemination", "probe-timeout", "l1")), 0.5))
    assert [s.node_id for s in reply.candidates] == ["a", first.to]
    # A reply arriving after the lookup finished is ignored.
    late = Received(second.to, ProbeReply(second.msg.request_id, record(second.to)))
    assert node.handle(late, 0.6) == []


def test_reply_about_a_different_node_is_not_used():
    node = sampling_node(sample_size=2)
    [probe] = probes(node.handle(Received("a", LookupRequest("l1")), 0.0))
    [reply] = replies(node.handle(Received(probe.to, ProbeReply(probe.msg.request_id, record("zz"))), 0.1))
    assert [s.node_id for s in reply.candidates] == ["a"]


def test_no_peers_to_probe_answers_immediately_with_own_status():
    for node in (sampling_node(neighbors=()), sampling_node(sample_size=1)):
        actions = node.handle(Received("a", LookupRequest("l1")), 0.0)
        assert not probes(actions)
        [reply] = replies(actions)
        assert [s.node_id for s in reply.candidates] == ["a"]


def test_probes_are_answered_with_this_workers_status():
    node = sampling_node()
    [answer] = node.handle(Received("b", ProbeRequest("p9")), 2.0)
    assert answer.to == "b"
    assert answer.msg == ProbeReply("p9", node.worker.status(2.0))


def test_own_freed_gpus_notify_the_scheduler():
    node = sampling_node()

    def notified(free, version):
        actions = node.handle(Received("a", StatusChanged(Status(spec("a"), (free,), 1, version, 0.0))), 0.0)
        return any(isinstance(a, Send) and isinstance(a.msg, CapacityAvailable) for a in actions)

    assert not notified(0, 1)
    assert notified(2, 2)


# ==================================================
# End to end
# ==================================================

CLUSTER = [NodeSpec(f"l40-{i}", [GpuGroup("L40", 2, 48)]) for i in range(1, 7)] + [
    NodeSpec(f"a100-{i}", [GpuGroup("A100", 2, 80)]) for i in range(1, 4)
]
IDS = [s.node_id for s in CLUSTER]


def run(seed=4):
    sim = Simulator(
        clusters.sampling(
            CLUSTER,
            neighborhoods=topology.ring_with_shortcuts(IDS, radius=1, seed=seed),
            placement=FirstFit(backfill=True),
            forwarding=RingForwarding(IDS),
            sample_size=2,
            retry_interval=0.25,
            seed=seed,
        ),
        latency=0.02,
    )
    # The default workload includes 64 GB jobs only A100 nodes can run.
    workload = poisson_workload(seed, 80, 4.0)
    for i, submission in enumerate(workload):
        sim.submit(submission.job, at=submission.at, node_id=IDS[i % len(IDS)])
    return workload, sim.run(until=2_000, until_jobs_done=True)


def test_sampling_cluster_completes_every_job_without_overbooking():
    workload, result = run()
    done = {o.metric.job_id for o in result.observations if isinstance(o.metric, JobCompleted)}
    assert done == {s.job.job_id for s in workload}
    assert result.messages["ProbeRequest"] > 0
    assert result.messages["ProbeRequest"] == result.messages["ProbeReply"]
    assert "StatusPush" not in result.messages  # no standing view to maintain

    usage = defaultdict(list)
    for execution in result.executions:
        usage[execution.node_id] += [(execution.start, execution.gpu_count), (execution.end, -execution.gpu_count)]
    for node_id, deltas in usage.items():
        in_use = 0
        for _, delta in sorted(deltas):
            in_use += delta
            assert 0 <= in_use <= 2, node_id


def test_sampling_runs_are_deterministic():
    _, first = run(seed=7)
    _, second = run(seed=7)
    assert first.observations == second.observations
    assert first.messages == second.messages


def test_from_config_builds_sampling_nodes():
    nodes = clusters.from_config(
        {
            "design": "sampling",
            "workers": [{"id": i, "gpus": [{"gpu_type": "A100", "count": 1, "vram_gb": 80}]} for i in "abc"],
            "sampling": {"sample_size": 2, "retry_interval": 0.25, "topology": "full"},
        }
    )
    assert [type(n.dissemination) for n in nodes] == [SamplingDissemination] * 3
    assert all(n.scheduler is not None and n.worker is not None for n in nodes)
    assert nodes[0].ctx.neighbors() == frozenset("bc")
    assert nodes[0].dissemination.sample_size == 2
