"""
Tests for crashed and restarted nodes: heartbeats, interruption, and loss.
"""

import pytest

from torchsched import clusters, topology
from torchsched.forwarding.ring import RingForwarding
from torchsched.placement.first_fit import FirstFit
from torchsched.runtime.http import load_runtime
from torchsched.runtime.sim import Simulator
from torchsched.types.metrics import JobCompleted, JobInterrupted, JobStarted
from torchsched.types.models import GpuGroup, Job, NodeSpec

# Single-GPU nodes, so where each job runs is easy to predict: a lookup
# lists this node's own worker first, then neighbors in id order.
IDS = ["a", "b", "c"]


def cluster(ids=IDS, *, forwarding=True):
    nodes = clusters.gossip(
        [NodeSpec(node_id, [GpuGroup("A100", 1, 80)]) for node_id in ids],
        neighborhoods=topology.full_mesh(ids),
        placement=FirstFit(backfill=True),
        forwarding=RingForwarding(ids) if forwarding else None,
        interval=0.5,
        max_staleness=3.0,
        retry_interval=0.25,
    )
    return Simulator(nodes, latency=0.02)


def job(job_id, runtime=10.0):
    return Job(job_id, 1, 40, {"A100": runtime})


def metrics(result, kind):
    return [o for o in result.observations if isinstance(o.metric, kind)]


def test_job_on_a_crashed_worker_is_rerun_elsewhere():
    sim = cluster()
    sim.submit(job("j1"), at=1.0, node_id="a")  # runs on a
    sim.submit(job("j2"), at=1.0, node_id="a")  # runs on b
    sim.crash("b", at=2.0)
    result = sim.run(until=100, until_jobs_done=True)

    [interrupted] = metrics(result, JobInterrupted)
    assert interrupted.metric == JobInterrupted("j2", "b", "no heartbeat")
    # Started on b just after t=1; heartbeats stopped with the crash.
    assert 4.0 < interrupted.time < 4.2
    assert [o.metric.worker for o in metrics(result, JobStarted) if o.metric.job_id == "j2"] == [
        "b",
        "c",
    ]
    assert {o.metric.job_id for o in metrics(result, JobCompleted)} == {"j1", "j2"}
    [lost_run] = [e for e in result.executions if e.interrupted]
    assert (lost_run.node_id, lost_run.job_id, lost_run.end) == ("b", "j2", 2.0)
    assert result.lost_jobs == set()
    assert result.dropped["Heartbeat"] == 0  # b sent none after crashing


def test_restart_is_noticed_from_the_new_incarnation_before_heartbeats_time_out():
    # Without forwarding, j3 waits at a, so a keeps looking up candidates.
    # (A scheduler with nothing waiting runs no lookups, so it relies on
    # heartbeats alone.)
    sim = cluster(["a", "b"], forwarding=False)
    sim.submit(job("j1"), at=1.0, node_id="a")  # runs on a
    sim.submit(job("j2"), at=1.0, node_id="a")  # runs on b
    sim.submit(job("j3"), at=1.0, node_id="a")  # waits
    sim.crash("b", at=2.0, restart_at=2.5)
    result = sim.run(until=100, until_jobs_done=True)

    [interrupted] = metrics(result, JobInterrupted)
    assert interrupted.metric == JobInterrupted("j2", "b", "worker restarted")
    assert interrupted.time < 3.0  # heartbeats would take until about 4
    assert sim.nodes["b"].ctx.incarnation == 2
    assert {o.metric.job_id for o in metrics(result, JobCompleted)} == {"j1", "j2", "j3"}


def test_long_jobs_stay_alive_on_heartbeats():
    sim = cluster()
    sim.submit(job("j1"), at=1.0, node_id="a")
    sim.submit(job("j2", runtime=30.0), at=1.0, node_id="a")  # on b
    result = sim.run(until=100, until_jobs_done=True)
    assert metrics(result, JobInterrupted) == []
    assert result.messages["Heartbeat"] >= 29  # about one per second from b


def crash_scheduler():
    sim = cluster()
    sim.submit(job("j1"), at=1.0, node_id="a")  # runs on a
    sim.submit(job("j2"), at=1.0, node_id="a")  # runs on b
    sim.submit(job("j3"), at=1.0, node_id="b")  # runs on c
    sim.crash("a", at=2.0)
    sim.submit(job("j4"), at=3.0, node_id="a")  # a is down
    return sim


def test_crashed_scheduler_loses_its_jobs():
    result = crash_scheduler().run(until=100, until_jobs_done=True)
    # j1 died with a's worker; j2 kept running on b, but its scheduler is
    # gone; j4 could not even be submitted.
    assert result.lost_jobs == {"j1", "j2", "j4"}
    assert {o.metric.job_id for o in metrics(result, JobCompleted)} == {"j3"}
    assert result.end_time < 100  # lost jobs do not keep the run going

    # Running on: b finishes j2 and resends JobFinished to the down a.
    result = crash_scheduler().run(until=30)
    assert result.dropped["JobFinished"] == 5


def test_a_node_can_crash_and_restart_repeatedly():
    sim = cluster()
    sim.crash("c", at=1.0, restart_at=2.0)
    sim.crash("c", at=3.0, restart_at=4.0)
    sim.submit(job("j1", runtime=1.0), at=6.0, node_id="c")
    result = sim.run(until=100, until_jobs_done=True)
    assert sim.nodes["c"].ctx.incarnation == 3
    assert [o.metric for o in metrics(result, JobCompleted)] == [JobCompleted("j1", "c")]


def test_crash_arguments_are_checked():
    sim = cluster()
    with pytest.raises(ValueError):
        sim.crash("zz", at=1.0)
    with pytest.raises(ValueError):
        sim.crash("a", at=2.0, restart_at=1.0)
    sim.nodes["a"].rebuild = None
    with pytest.raises(ValueError):
        sim.crash("a", at=1.0, restart_at=2.0)
    sim.crash("a", at=1.0)  # crashing for good needs no rebuild


def test_http_nodes_take_their_boot_time_as_incarnation():
    config = {
        "design": "gossip",
        "workers": [{"id": i, "gpus": [{"gpu_type": "A100", "count": 1, "vram_gb": 80}]} for i in "ab"],
        "gossip": {"interval": 0.5, "max_staleness": 3.0, "retry_interval": 0.25},
        "addresses": {"a": "http://a:8000", "b": "http://b:8000"},
    }
    first = load_runtime(config, "a").node.ctx.incarnation
    assert first > 1_700_000_000_000  # milliseconds since the epoch
    assert load_runtime(config, "a").node.ctx.incarnation >= first
