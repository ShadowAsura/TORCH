"""
The three demonstration scenarios from the legacy simulator, for the report.

1. Stale views send two schedulers after the same GPU. The worker grants
   one reservation and rejects the other, and the second job starts once
   the GPU frees up.
2. A worker crashes mid-job. Its scheduler notices the missing heartbeats
   and reruns the job on another worker.
3. Status messages are lost for a while, hiding remote capacity. Ring
   forwarding still gets the job to a node that can run it, and views fill
   in again once status messages get through.
"""

from torchsched import clusters, topology
from torchsched.forwarding.ring import RingForwarding
from torchsched.placement.first_fit import FirstFit
from torchsched.runtime.sim import MessageLoss, Simulator
from torchsched.types.metrics import (
    JobCompleted,
    JobInterrupted,
    JobStarted,
    ReservationRejected,
)
from torchsched.types.models import GpuGroup, Job, NodeSpec

# Nodes a and b have L40s; only c has an A100. Every job here needs an A100.
IDS = ["a", "b", "c"]
SPECS = [
    NodeSpec("a", [GpuGroup("L40", 1, 48)]),
    NodeSpec("b", [GpuGroup("L40", 1, 48)]),
    NodeSpec("c", [GpuGroup("A100", 1, 80)]),
]


def simulator(specs=SPECS, *, losses=()):
    ids = [spec.node_id for spec in specs]
    nodes = clusters.gossip(
        specs,
        neighborhoods=topology.full_mesh(ids),
        placement=FirstFit(backfill=True),
        forwarding=RingForwarding(ids),
        interval=0.5,
        max_staleness=3.0,
        retry_interval=0.25,
    )
    return Simulator(nodes, latency=0.02, losses=losses)


def a100_job(job_id, runtime=5.0):
    return Job(job_id, 1, 64, {"A100": runtime})


def observed(result, kind):
    return [o for o in result.observations if isinstance(o.metric, kind)]


def test_stale_views_compete_and_the_worker_settles_it():
    sim = simulator()
    # Submitted at the same moment to two schedulers whose views both show
    # c's A100 free.
    sim.submit(a100_job("j1"), at=2.0, node_id="a")
    sim.submit(a100_job("j2"), at=2.0, node_id="b")
    result = sim.run(until=100, until_jobs_done=True)

    [rejected] = observed(result, ReservationRejected)
    assert rejected.metric.worker == "c" and rejected.time < 2.1
    starts = {o.metric.job_id: o.time for o in observed(result, JobStarted)}
    first, second = sorted(starts, key=starts.get)
    assert rejected.metric.job_id == second
    # The second job starts only after the first one's GPU is released.
    runs = {e.job_id: e for e in result.executions}
    assert runs[second].start >= runs[first].end
    assert {o.metric.job_id for o in observed(result, JobCompleted)} == {"j1", "j2"}


def test_crashed_worker_job_is_rerun_on_another_worker():
    specs = [*SPECS, NodeSpec("d", [GpuGroup("A100", 1, 80)])]
    sim = simulator(specs)
    # By t=3 gossip has filled a's view, so a's scheduler reserves c's A100
    # itself. (Had the job been forwarded to c, c's scheduler would have
    # owned it and been lost in the crash along with the run.)
    sim.submit(a100_job("j1", runtime=10.0), at=3.0, node_id="a")  # runs on c
    sim.crash("c", at=5.0)
    result = sim.run(until=100, until_jobs_done=True)

    [interrupted] = observed(result, JobInterrupted)
    assert interrupted.metric.worker == "c"
    workers = [o.metric.worker for o in observed(result, JobStarted)]
    assert workers == ["c", "d"]
    assert [o.metric for o in observed(result, JobCompleted)] == [JobCompleted("j1", "d")]
    assert [(e.node_id, e.interrupted) for e in result.executions] == [
        ("c", True),
        ("d", False),
    ]


def test_lost_status_messages_are_bridged_by_forwarding():
    sim = simulator(losses=[MessageLoss(end=5.0, kinds={"StatusPush"})])
    sim.submit(a100_job("j1"), at=1.0, node_id="a")
    result = sim.run(until=4.0)

    # a's view is empty, so it forwards the job; c runs it on its own A100.
    [started] = observed(result, JobStarted)
    assert started.metric.worker == "c" and started.time < 1.1
    assert sim.nodes["a"].dissemination.view == {}
    assert result.dropped["StatusPush"] > 0

    # Once status messages get through again, views fill back in.
    sim.run(until=8.0)
    assert set(sim.nodes["a"].dissemination.view) == {"b", "c"}
