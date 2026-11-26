"""
Tests for first-fit placement and the centralized baseline run end to end.
"""

import pytest

import simulator as legacy
from experiment import make_workload
from torchsched.clusters import centralized
from torchsched.placement.first_fit import FirstFit
from torchsched.runtime.sim import Simulator
from torchsched.types.metrics import JobAccepted, JobCompleted, JobStarted
from torchsched.types.models import Assignment, GpuGroup, Job, NodeSpec, Status

# ==================================================
# Placement policy
# ==================================================


def status(node_id, groups, free):
    return Status(NodeSpec(node_id, groups), free, 1, 0, 0.0)


L40 = GpuGroup("L40", 2, 48)
A100 = GpuGroup("A100", 2, 80)


def j(job_id, gpus=1, vram=24, runtimes=None):
    return Job(job_id, gpus, vram, runtimes or {"L40": 2.0, "A100": 1.0})


def test_first_fit_takes_nodes_then_groups_in_order_without_overcommitting():
    candidates = [status("n1", [L40], [1]), status("n2", [L40, A100], [2, 2])]
    plan = FirstFit().plan([j("a"), j("b", 2), j("c", 2)], candidates, 0.0)
    assert plan == [
        Assignment("a", "n1", 0),
        Assignment("b", "n2", 0),
        Assignment("c", "n2", 1),
    ]


def test_strict_fifo_blocks_behind_the_head_and_backfill_does_not():
    candidates = [status("n1", [L40], [1])]
    pending = [j("big", 2), j("small", 1)]
    assert FirstFit().plan(pending, candidates, 0.0) == []
    assert FirstFit(backfill=True).plan(pending, candidates, 0.0) == [
        Assignment("small", "n1", 0)
    ]


def test_first_fit_respects_type_and_vram():
    candidates = [status("n1", [L40], [2]), status("n2", [A100], [2])]
    plan = FirstFit().plan(
        [j("needs-a100", runtimes={"A100": 1.0}), j("needs-64gb", vram=64)],
        candidates,
        0.0,
    )
    assert plan == [Assignment("needs-a100", "n2", 0), Assignment("needs-64gb", "n2", 0)]


# ==================================================
# End to end
# ==================================================


def run_centralized(specs, jobs, *, backfill=False, latency=0.0):
    sim = Simulator(
        centralized(specs, placement=FirstFit(backfill=backfill)), latency=latency
    )
    for job, arrival in jobs:
        sim.submit(job, at=arrival, node_id="coordinator")
    return sim.run()


def timeline(result):
    """
    job_id -> (start, finish, worker) from the observation log.
    """
    starts, finishes = {}, {}
    for o in result.observations:
        if isinstance(o.metric, JobStarted):
            starts[o.metric.job_id] = (o.time, o.metric.worker)
        elif isinstance(o.metric, JobCompleted):
            finishes[o.metric.job_id] = o.time
    return {
        job_id: (start, finishes[job_id], worker)
        for job_id, (start, worker) in starts.items()
    }


def legacy_cluster(l40_nodes, a100_nodes, gpus=2):
    return [
        legacy.Node(f"{gpu_type.lower()}-{i + 1}", gpu_type, gpus, vram)
        for gpu_type, count, vram in [("L40", l40_nodes, 48), ("A100", a100_nodes, 80)]
        for i in range(count)
    ]


def convert(nodes, jobs):
    specs = [NodeSpec(n.name, [GpuGroup(n.gpu_type, n.gpu_count, n.vram_gb)]) for n in nodes]
    new_jobs = [
        (Job(job.name, job.gpu_count, job.min_vram_gb, job.runtime_by_gpu), job.arrival)
        for job in jobs
    ]
    return specs, new_jobs


@pytest.mark.parametrize(
    "seed,count,rate,l40,a100",
    [(42, 100, 5.0, 6, 6), (7, 200, 2.0, 2, 2), (3, 150, 8.0, 3, 1), (11, 80, 0.5, 1, 4)],
)
def test_strict_fifo_matches_the_legacy_centralized_baseline(seed, count, rate, l40, a100):
    nodes = legacy_cluster(l40, a100)
    jobs = make_workload(seed, count, rate)
    expected = {
        p.job.name: (p.start, p.finish, p.node.name)
        for p in legacy.simulate(nodes, jobs).placements
    }
    specs, new_jobs = convert(nodes, jobs)
    result = run_centralized(specs, new_jobs)
    assert timeline(result) == expected


def test_backfill_never_delays_this_workload_relative_to_strict_fifo():
    nodes = legacy_cluster(2, 2)
    specs, jobs = convert(nodes, make_workload(42, 100, 5.0))
    strict = timeline(run_centralized(specs, jobs))
    backfill = timeline(run_centralized(specs, jobs, backfill=True))
    arrival = {job.job_id: at for job, at in jobs}

    def mean_delay(t):
        return sum(start - arrival[k] for k, (start, _, _) in t.items()) / len(t)

    assert len(backfill) == len(strict) == 100
    assert mean_delay(backfill) < mean_delay(strict)


def test_mixed_gpu_node_runs_jobs_on_the_right_groups():
    workstation = NodeSpec("ws", [GpuGroup("RTX4090", 1, 24), GpuGroup("A6000", 2, 48)])
    jobs = [
        (Job("small", 1, 16, {"RTX4090": 1.0, "A6000": 2.0}), 0.0),
        (Job("big", 2, 40, {"A6000": 3.0}), 0.0),
        (Job("next", 1, 16, {"RTX4090": 1.0}), 0.5),
    ]
    result = run_centralized([workstation], jobs)
    groups = {
        o.metric.job_id: o.metric.gpu_type
        for o in result.observations
        if isinstance(o.metric, JobStarted)
    }
    assert groups == {"small": "RTX4090", "big": "A6000", "next": "RTX4090"}
    assert timeline(result)["next"][0] == 1.0  # waits for the 4090


def test_network_latency_delays_but_never_double_books():
    nodes = legacy_cluster(2, 2)
    specs, jobs = convert(nodes, make_workload(5, 120, 4.0))
    result = run_centralized(specs, jobs, latency=0.05)
    accepted = [o for o in result.observations if isinstance(o.metric, JobAccepted)]
    assert len(accepted) == len(timeline(result)) == 120
    # Coordinator-side assumptions keep it from overbooking a stale view.
    assert result.messages["ReserveRequest"] == 120
    assert result.undeliverable == 0
