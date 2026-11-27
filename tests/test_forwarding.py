"""
Tests for job forwarding between schedulers.
"""

import random
from collections import defaultdict

import pytest

from torchsched import clusters, topology
from torchsched.forwarding.ring import RingForwarding
from torchsched.membership.static import StaticMembership
from torchsched.node import Node, NodeContext
from torchsched.placement.first_fit import FirstFit
from torchsched.runtime.sim import Simulator
from torchsched.scheduler import Scheduler
from torchsched.types.actions import Observe, Send, SetTimer
from torchsched.types.events import JobSubmitted, Received, TimerFired
from torchsched.types.messages import ForwardJob, LookupReply, LookupRequest, ReserveRequest
from torchsched.types.metrics import JobAccepted, JobCompleted, JobForwarded, PlacementRetried
from torchsched.types.models import GpuGroup, Job, NodeSpec, Status

# ==================================================
# Ring policy
# ==================================================

RING = ["a", "b", "c"]


def test_ring_forwarding_follows_the_ring_and_ends_laps():
    ring = RingForwarding(RING)
    everyone = frozenset(RING)
    job = Job("j", 1, 1, {"A100": 1.0})
    assert ring.next_hop(job, ["a"], everyone - {"a"}) == "b"
    assert ring.next_hop(job, ["b", "c"], everyone - {"c"}) == "a"
    # The next node already tried the job on this lap.
    assert ring.next_hop(job, ["a", "b", "c"], everyone - {"c"}) is None
    # The next node is not a neighbor.
    assert ring.next_hop(job, ["a"], frozenset({"c"})) is None
    # A node outside the ring has no successor.
    assert ring.next_hop(job, ["zz"], everyone) is None
    with pytest.raises(ValueError):
        RingForwarding(["a", "a"])


# ==================================================
# Scheduler behavior, driven by hand
# ==================================================


class NoDissemination:
    """
    Swallows lookups; the test answers them itself.
    """

    name = "dissemination"
    accepts = frozenset({LookupRequest})

    def handle(self, event, now):
        return []


def record(node_id, free, count=2):
    return Status(NodeSpec(node_id, [GpuGroup("A100", count, 80)]), (free,), 1, 0, 0.0)


def job(job_id, gpus=1):
    return Job(job_id, gpus, 40, {"A100": 1.0})


class Harness:
    def __init__(self, node_id="a", *, forwarding=True):
        ctx = NodeContext(NodeSpec(node_id), incarnation=1, seed=0)
        self.node = Node(
            ctx,
            membership=StaticMembership(set(RING) - {node_id}),
            dissemination=NoDissemination(),
            scheduler=Scheduler(
                ctx,
                placement=FirstFit(),
                forwarding=RingForwarding(RING) if forwarding else None,
                retry_interval=0.25,
            ),
        )
        self.lookup_id = None

    def feed(self, event):
        actions = self.node.handle(event, 0.0)
        for action in actions:
            if isinstance(action, Send) and isinstance(action.msg, LookupRequest):
                self.lookup_id = action.msg.lookup_id
        return actions

    def answer(self, *candidates):
        lookup_id, self.lookup_id = self.lookup_id, None
        return self.feed(Received(self.node.node_id, LookupReply(lookup_id, candidates)))


def sent(actions, kind):
    return [a for a in actions if isinstance(a, Send) and isinstance(a.msg, kind)]


def observed(actions, kind):
    return [a.metric for a in actions if isinstance(a, Observe) and isinstance(a.metric, kind)]


def test_unplaceable_head_is_forwarded_and_unblocks_the_queue():
    h = Harness("a")
    h.feed(JobSubmitted(job("big", gpus=2)))
    h.feed(JobSubmitted(job("small", gpus=1)))
    # Only one GPU free in the view: strict FIFO places nothing this round.
    actions = h.answer(record("b", free=1))
    [forward] = sent(actions, ForwardJob)
    assert forward.to == "b"
    assert forward.msg == ForwardJob(job("big", gpus=2), ("a",))
    assert observed(actions, JobForwarded) == [JobForwarded("big", "b")]
    # "small" fits, so it stays, and forwarding the head starts a new round.
    assert not [f for f in sent(actions, ForwardJob) if f.msg.job.job_id == "small"]
    assert h.lookup_id is not None
    [reserve] = sent(h.answer(record("b", free=1)), ReserveRequest)
    assert reserve.msg.job.job_id == "small"


def test_forwarded_job_is_owned_by_the_receiver_and_continues_the_lap():
    h = Harness("b")
    actions = h.feed(Received("a", ForwardJob(job("big", gpus=2), ("a",))))
    assert not observed(actions, JobAccepted)  # accepted once, at its origin
    assert h.lookup_id is not None
    [forward] = sent(h.answer(record("b", free=0)), ForwardJob)
    assert forward.to == "c"
    assert forward.msg.visited == ("a", "b")
    # A duplicate of a job this scheduler owns is ignored.
    h2 = Harness("b")
    h2.feed(Received("a", ForwardJob(job("x"), ("a",))))
    assert h2.feed(Received("c", ForwardJob(job("x"), ("c",)))) == []


def test_lap_end_waits_for_the_retry_timer_then_starts_a_new_lap():
    h = Harness("c")
    h.feed(Received("b", ForwardJob(job("big", gpus=2), ("a", "b"))))
    actions = h.answer(record("c", free=0))
    # "a" already tried it this lap, so the job stays and waits.
    assert not sent(actions, ForwardJob)
    assert SetTimer(("scheduler", "retry"), 0.25) in actions

    actions = h.feed(TimerFired(("scheduler", "retry")))
    assert observed(actions, PlacementRetried)
    [forward] = sent(h.answer(record("c", free=0)), ForwardJob)
    assert forward.to == "a"
    assert forward.msg.visited == ("c",)  # a fresh lap from here


def test_without_a_forwarding_policy_jobs_stay_put():
    h = Harness("a", forwarding=False)
    h.feed(JobSubmitted(job("big", gpus=2)))
    assert not sent(h.answer(record("b", free=0)), ForwardJob)


# ==================================================
# End to end
# ==================================================

# A ring where each L40 node's neighborhood may contain no A100 at all.
CLUSTER = [NodeSpec(f"l40-{i}", [GpuGroup("L40", 2, 48)]) for i in range(1, 7)] + [
    NodeSpec(f"a100-{i}", [GpuGroup("A100", 2, 80)]) for i in range(1, 3)
]
IDS = [spec.node_id for spec in CLUSTER]


def workload(count=60, seed=2):
    """
    A mix that includes 64 GB jobs only the A100 nodes can run.
    """
    rng = random.Random(seed)
    at, jobs = 0.0, []
    for i in range(count):
        at += rng.expovariate(2.0)
        runtime = rng.uniform(1, 4)
        vram = rng.choice([24, 24, 64])
        runtimes = {"A100": runtime * 0.75} if vram == 64 else {"L40": runtime, "A100": runtime * 0.75}
        jobs.append((Job(f"j{i}", 1, vram, runtimes), at))
    return jobs


def run(jobs, *, forwarding, seed=5, until=500):
    sim = Simulator(
        clusters.gossip(
            CLUSTER,
            neighborhoods=topology.ring_with_shortcuts(IDS, radius=1, seed=seed),
            placement=FirstFit(),
            forwarding=RingForwarding(IDS) if forwarding else None,
            interval=0.5,
            max_staleness=3.0,
            retry_interval=0.25,
            seed=seed,
        ),
        latency=0.02,
    )
    for i, (j, at) in enumerate(jobs):
        sim.submit(j, at=at, node_id=IDS[i % len(IDS)])
    return sim.run(until=until, until_jobs_done=True)


def completed(result):
    return {o.metric.job_id for o in result.observations if isinstance(o.metric, JobCompleted)}


def test_forwarding_lets_every_job_reach_a_neighborhood_that_can_run_it():
    jobs = workload()
    stuck = run(jobs, forwarding=False)
    assert completed(stuck) < {j.job_id for j, _ in jobs}  # some never run

    result = run(jobs, forwarding=True)
    assert completed(result) == {j.job_id for j, _ in jobs}
    assert result.messages["ForwardJob"] > 0
    accepted = [o for o in result.observations if isinstance(o.metric, JobAccepted)]
    assert len(accepted) == len(jobs)  # forwarding never double counts arrivals

    # 64 GB jobs ran only on A100 nodes, and no worker was ever overbooked.
    big = {j.job_id for j, _ in jobs if j.min_vram_gb == 64}
    usage = defaultdict(list)
    for run_ in result.executions:
        if run_.job_id in big:
            assert run_.node_id.startswith("a100")
        usage[run_.node_id] += [(run_.start, run_.gpu_count), (run_.end, -run_.gpu_count)]
    for node_id, deltas in usage.items():
        in_use = 0
        for _, delta in sorted(deltas):
            in_use += delta
            assert 0 <= in_use <= 2, node_id


def test_forwarding_runs_are_deterministic():
    jobs = workload()
    first, second = run(jobs, forwarding=True), run(jobs, forwarding=True)
    assert first.observations == second.observations
    assert first.messages == second.messages
