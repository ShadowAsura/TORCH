"""
Tests for lost messages: resending, giving up, and loss injection.
"""

from collections import Counter, defaultdict

import pytest

from torchsched import clusters, topology
from torchsched.forwarding.ring import RingForwarding
from torchsched.membership.static import StaticMembership
from torchsched.node import Node, NodeContext
from torchsched.placement.first_fit import FirstFit
from torchsched.retransmit import Retransmitter
from torchsched.runtime.sim import Simulator
from torchsched.scenarios import MessageLoss
from torchsched.scheduler import Scheduler
from torchsched.types.actions import CancelTimer, Observe, Send, SetTimer
from torchsched.types.events import JobSubmitted, Received, TimerFired
from torchsched.types.messages import (
    ForwardAck,
    ForwardJob,
    LookupReply,
    LookupRequest,
)
from torchsched.types.metrics import JobCompleted, JobInterrupted, RequestAbandoned
from torchsched.types.models import GpuGroup, Job, NodeSpec, Status
from torchsched.workloads import poisson_workload

# ==================================================
# Retransmitter
# ==================================================


def test_retransmitter_resends_until_acknowledged_or_out_of_sends():
    outbox = Retransmitter(timeout=0.5, max_sends=3)
    msg = ForwardAck("j1", "h1")
    assert outbox.send(("forward", "j1"), "b", msg) == [
        Send("b", msg),
        SetTimer(("resend", "forward", "j1"), 0.5),
    ]
    assert outbox.timed_out(("forward", "j1")) == (
        [Send("b", msg), SetTimer(("resend", "forward", "j1"), 0.5)],
        None,
    )
    actions, abandoned = outbox.timed_out(("forward", "j1"))
    assert len(actions) == 2 and abandoned is None  # third and last send
    actions, abandoned = outbox.timed_out(("forward", "j1"))
    assert actions == [] and abandoned.to == "b" and abandoned.msg == msg
    assert ("forward", "j1") not in outbox
    assert outbox.timed_out(("forward", "j1")) == ([], None)

    outbox.send(("forward", "j2"), "c", msg)
    assert outbox.acknowledged(("forward", "j2")) == [CancelTimer(("resend", "forward", "j2"))]
    assert outbox.acknowledged(("forward", "j2")) == []


# ==================================================
# Loss injection
# ==================================================


def test_message_loss_matching_and_validation():
    loss = MessageLoss(start=1.0, end=2.0, kinds={"StatusPush"}, receivers=["b"])
    assert loss.matches(1.0, "a", "b", "StatusPush")
    assert not loss.matches(2.0, "a", "b", "StatusPush")  # end is exclusive
    assert not loss.matches(1.5, "a", "c", "StatusPush")
    assert not loss.matches(1.5, "a", "b", "ProbeReply")
    assert MessageLoss().matches(99.0, "x", "y", "Anything")
    with pytest.raises(ValueError):
        MessageLoss(probability=1.5)
    with pytest.raises(TypeError):
        MessageLoss(kinds="StatusPush")


# ==================================================
# Scenarios: one coordinator, one worker
# ==================================================

WORKER = NodeSpec("w", [GpuGroup("A100", 1, 80)])


def centralized_run(*losses, runtime=10.0):
    sim = Simulator(
        clusters.centralized([WORKER], placement=FirstFit()),
        latency=0.02,
        losses=losses,
    )
    sim.submit(Job("j1", 1, 40, {"A100": runtime}), at=1.0, node_id="coordinator")
    return sim.run(until=200, until_jobs_done=True)


def metrics(result, kind):
    return [o.metric for o in result.observations if isinstance(o.metric, kind)]


def test_lost_reserve_reply_is_resent_and_granted_once():
    result = centralized_run(MessageLoss(end=1.5, kinds={"ReserveReply"}))
    assert result.dropped == Counter({"ReserveReply": 1})
    assert result.messages["ReserveRequest"] == 2
    # The worker recognized the resent request and repeated its grant.
    assert len(result.executions) == 1
    assert metrics(result, JobCompleted) == [JobCompleted("j1", "w")]


def test_unconfirmed_start_is_abandoned_but_its_finish_still_counts():
    result = centralized_run(MessageLoss(kinds={"StartReply"}))
    assert result.messages["StartRequest"] == 5  # the default max_sends
    assert metrics(result, RequestAbandoned) == [RequestAbandoned("j1", "w", "StartRequest")]
    # The worker had started it on the first request; repeats did not rerun it.
    assert len(result.executions) == 1
    assert metrics(result, JobCompleted) == [JobCompleted("j1", "w")]


def test_lost_job_finished_is_resent_until_acknowledged():
    result = centralized_run(MessageLoss(end=11.5, kinds={"JobFinished"}))
    assert result.dropped == Counter({"JobFinished": 1})
    assert result.messages["JobFinished"] == 2
    assert result.messages["FinishedAck"] == 1
    assert metrics(result, JobCompleted) == [JobCompleted("j1", "w")]


def test_worker_stops_resending_when_acks_never_arrive():
    sim = Simulator(
        clusters.centralized([WORKER], placement=FirstFit()),
        latency=0.02,
        losses=[MessageLoss(kinds={"FinishedAck"})],
    )
    sim.submit(Job("j1", 1, 40, {"A100": 1.0}), at=1.0, node_id="coordinator")
    result = sim.run(until=100)  # keep going after the job completes
    assert result.messages["JobFinished"] == 5  # the default max_sends
    assert result.messages["FinishedAck"] == 5  # duplicates are acknowledged too
    assert metrics(result, JobCompleted) == [JobCompleted("j1", "w")]


# ==================================================
# Forwarding handoffs
# ==================================================


class NoDissemination:
    name = "dissemination"
    accepts = frozenset({LookupRequest})

    def handle(self, event, now):
        return []


def forwarding_scheduler(node_id="a"):
    ctx = NodeContext(NodeSpec(node_id), incarnation=1, seed=0)
    node = Node(
        ctx,
        membership=StaticMembership({"a", "b", "c"} - {node_id}),
        dissemination=NoDissemination(),
        scheduler=Scheduler(
            ctx, placement=FirstFit(), forwarding=RingForwarding(["a", "b", "c"]), max_sends=2
        ),
    )
    return node


def forward_big_job(node):
    """
    Submit a job nothing in the (empty) view can run, so it is forwarded.
    """
    big = Job("big", 2, 40, {"A100": 1.0})
    [lookup] = node.handle(JobSubmitted(big), 0.0)[1:]
    full = Status(NodeSpec("a", [GpuGroup("A100", 2, 80)]), (0,), 1, 0, 0.0)
    actions = node.handle(Received("a", LookupReply(lookup.msg.lookup_id, (full,))), 0.0)
    [forward] = [a for a in actions if isinstance(a, Send) and isinstance(a.msg, ForwardJob)]
    return big, forward


def test_forwarded_job_is_kept_until_acknowledged():
    node = forwarding_scheduler()
    _, forward = forward_big_job(node)
    assert forward.to == "b"
    # Unacknowledged: the same handoff is sent again.
    resend = ("scheduler", "resend", "forward", forward.msg.handoff_id)
    actions = node.handle(TimerFired(resend), 1.0)
    assert Send("b", forward.msg) in actions
    # Acknowledged: the job is no longer this scheduler's.
    ack = ForwardAck("big", forward.msg.handoff_id)
    assert node.handle(Received("b", ack), 1.1) == [CancelTimer(resend)]
    assert node.scheduler._jobs == {}


def test_resent_handoff_is_acknowledged_but_not_accepted_twice():
    receiver = forwarding_scheduler("b")
    big = Job("big", 2, 40, {"A100": 1.0})
    first = receiver.handle(Received("a", ForwardJob(big, ("a",), "h1")), 0.0)
    assert any(isinstance(a, Send) and isinstance(a.msg, LookupRequest) for a in first)
    # Forwarded onward, so this scheduler no longer holds it...
    del receiver.scheduler._jobs["big"]
    # ...and the resend (its ack was lost) must not bring it back.
    assert receiver.handle(Received("a", ForwardJob(big, ("a",), "h1")), 1.0) == [
        Send("a", ForwardAck("big", "h1"))
    ]
    assert receiver.scheduler._jobs == {}


def test_unacknowledged_forward_is_abandoned_and_the_job_queued_here():
    node = forwarding_scheduler()
    _, forward = forward_big_job(node)
    resend = ("scheduler", "resend", "forward", forward.msg.handoff_id)
    node.handle(TimerFired(resend), 1.0)
    actions = node.handle(TimerFired(resend), 2.0)
    assert Observe(RequestAbandoned("big", "b", "ForwardJob")) in actions
    assert any(isinstance(a, Send) and isinstance(a.msg, LookupRequest) for a in actions)


def test_job_that_laps_back_before_its_ack_is_taken_back():
    node = forwarding_scheduler()
    big, forward = forward_big_job(node)
    actions = node.handle(Received("c", ForwardJob(big, ("a", "b", "c"), "c:1:handoff-1")), 0.5)
    assert Send("c", ForwardAck("big", "c:1:handoff-1")) in actions
    assert CancelTimer(("scheduler", "resend", "forward", forward.msg.handoff_id)) in actions
    assert node.scheduler._jobs["big"].visited == ("a", "b", "c", "a")


# ==================================================
# End to end with random loss
# ==================================================

CLUSTER = [NodeSpec(f"l40-{i}", [GpuGroup("L40", 2, 48)]) for i in range(1, 7)] + [
    NodeSpec(f"a100-{i}", [GpuGroup("A100", 2, 80)]) for i in range(1, 4)
]
IDS = [s.node_id for s in CLUSTER]


def lossy_run(seed=3, probability=0.1):
    sim = Simulator(
        clusters.gossip(
            CLUSTER,
            neighborhoods=topology.ring_with_shortcuts(IDS, radius=1, seed=seed),
            placement=FirstFit(backfill=True),
            forwarding=RingForwarding(IDS),
            interval=0.5,
            max_staleness=3.0,
            retry_interval=0.25,
            seed=seed,
        ),
        latency=0.02,
        losses=[MessageLoss(probability=probability)],
        seed=seed,
    )
    # Includes 64 GB jobs only A100 nodes can run, so forwarding is needed.
    workload = poisson_workload(seed, 80, 4.0)
    for i, submission in enumerate(workload):
        sim.submit(submission.job, at=submission.at, node_id=IDS[i % len(IDS)])
    return workload, sim.run(until=5_000, until_jobs_done=True)


@pytest.mark.parametrize("seed, probability", [(3, 0.1), (5, 0.2)])
def test_every_job_completes_despite_random_loss(seed, probability):
    workload, result = lossy_run(seed, probability)
    assert sum(result.dropped.values()) > 100
    completions = Counter(m.job_id for m in metrics(result, JobCompleted))
    assert set(completions) == {s.job.job_id for s in workload}
    # Only a wrong guess that a message or a run was lost (giving up on a
    # request, or missing heartbeats) can duplicate a job.
    duplicated = {job_id for job_id, count in completions.items() if count > 1}
    guessed = {m.job_id for m in metrics(result, RequestAbandoned)} | {
        m.job_id for m in metrics(result, JobInterrupted)
    }
    assert duplicated <= guessed

    # Duplicate runs are allowed, but never more GPUs than a worker has.
    usage = defaultdict(list)
    for execution in result.executions:
        usage[execution.node_id] += [(execution.start, execution.gpu_count), (execution.end, -execution.gpu_count)]
    for node_id, deltas in usage.items():
        in_use = 0
        for _, delta in sorted(deltas):
            in_use += delta
            assert 0 <= in_use <= 2, node_id


def test_lossy_runs_are_deterministic():
    _, first = lossy_run(seed=8)
    _, second = lossy_run(seed=8)
    assert first.observations == second.observations
    assert first.dropped == second.dropped
