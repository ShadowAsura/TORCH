"""
Tests for fault scenarios: parsing, resolving random crashes, and running
them in the simulator.
"""

import argparse
import json
from pathlib import Path
from random import Random

import pytest

from experiments.compare_with_legacy import cluster
from torchsched import clusters, topology
from torchsched.forwarding.ring import RingForwarding
from torchsched.placement.first_fit import FirstFit
from torchsched.runtime.sim import Simulator
from torchsched.scenarios import (
    Crash,
    MessageLoss,
    Partition,
    RandomCrashes,
    Scenario,
    is_dropped,
    load_scenario,
    loss_from_data,
    loss_to_data,
    scenario_argument,
    scenario_from_data,
)
from torchsched.types.metrics import JobCompleted, JobInterrupted, JobStarted
from torchsched.types.models import GpuGroup, Job, NodeSpec

EXAMPLES = sorted(
    [*Path("deploy/scenarios").glob("*.json"), *Path("experiments/scenarios").glob("*.json")]
)


def test_scenario_parses_every_section():
    scenario = scenario_from_data(
        {
            "crashes": [
                {"node": "b", "at": 9},
                {"node": "a", "at": 1, "restart_at": 4},
                {"node": "a", "at": 6},
            ],
            "random_crashes": {"count": 2, "downtime": 5, "window": [0, 20]},
            "partitions": [{"nodes": ["c", "d"], "at": 2, "heal_at": 8}],
            "losses": [{"probability": 0.5, "kinds": ["StatusPush"], "end": 3}],
        }
    )
    assert scenario.crashes == (Crash("a", 1, 4), Crash("a", 6), Crash("b", 9))
    assert scenario.random_crashes == RandomCrashes(2, 5.0, (0.0, 20.0))
    assert scenario.partitions == (Partition(frozenset({"c", "d"}), 2, 8),)
    assert scenario.losses == (MessageLoss(end=3, probability=0.5, kinds={"StatusPush"}),)
    assert scenario.crashed == {"a", "b"}
    assert scenario.partitioned == {"c", "d"}
    assert scenario.nodes == {"a", "b", "c", "d"}
    assert scenario.down_at_end() == {"a", "b"}
    assert scenario_from_data({}) == Scenario()


@pytest.mark.parametrize(
    ("data", "message"),
    [
        ({"crash": []}, "unknown scenario fields"),
        ({"crashes": [{"node": "a", "at": 1, "until": 2}]}, "unknown crash fields"),
        ({"crashes": [{"node": "a"}]}, "crash needs at"),
        ({"crashes": [{"node": "a", "at": -1}]}, "at must be nonnegative"),
        ({"crashes": [{"node": "a", "at": 2, "restart_at": 2}]}, "restart_at must be after at"),
        (
            {"crashes": [{"node": "a", "at": 1, "restart_at": 5}, {"node": "a", "at": 3}]},
            "while already down",
        ),
        ({"crashes": [{"node": "a", "at": 1}, {"node": "a", "at": 3}]}, "while already down"),
        ({"random_crashes": {"count": 1, "downtime": 0}}, "downtime must be positive"),
        ({"random_crashes": {"count": 1, "window": [5, 1]}}, "window must be"),
        ({"partitions": [{"nodes": [], "at": 1}]}, "at least one node"),
        ({"partitions": [{"nodes": ["a"], "at": 3, "heal_at": 1}]}, "heal_at must be after"),
        (
            {
                "partitions": [
                    {"nodes": ["a", "b"], "at": 1, "heal_at": 5},
                    {"nodes": ["b"], "at": 4},
                ]
            },
            "in two partitions",
        ),
        ({"losses": [{"probability": 2}]}, "probability"),
        ({"losses": [{"start": 3, "end": 1}]}, "end must be after"),
    ],
)
def test_bad_scenarios_are_rejected(data, message):
    with pytest.raises(ValueError, match=message):
        scenario_from_data(data)


def test_names_must_be_lists():
    with pytest.raises(TypeError):
        scenario_from_data({"losses": [{"kinds": "StatusPush"}]})


def test_bad_scenario_files_are_usage_errors(tmp_path):
    with pytest.raises(argparse.ArgumentTypeError):
        scenario_argument(str(tmp_path / "missing.json"))
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"crashes": [{"node": "a"}]}))
    with pytest.raises(argparse.ArgumentTypeError, match="crash needs at"):
        scenario_argument(str(bad))


@pytest.mark.parametrize("path", EXAMPLES, ids=[f"{p.parent.name}/{p.stem}" for p in EXAMPLES])
def test_examples_name_nodes_in_both_clusters(path):
    scenario = load_scenario(path)
    deployed = json.loads(Path("deploy/cluster.json").read_text())
    assert scenario.nodes <= {worker["id"] for worker in deployed["workers"]}
    assert scenario.nodes <= {spec.node_id for spec in cluster(6, 6)}


def test_losses_round_trip_through_data():
    loss = MessageLoss(start=1, end=2, probability=0.3, senders={"a"}, receivers={"b", "c"})
    assert loss_from_data(json.loads(json.dumps(loss_to_data(loss)))) == loss
    assert loss_to_data(MessageLoss()) == {"start": 0.0, "probability": 1.0}


def test_random_crashes_resolve_per_seed():
    scenario = Scenario((Crash("a", 1.0),), RandomCrashes(2, downtime=5, window=(10, 20)))
    nodes = ["a", "b", "c", "d", "e"]
    first = scenario.resolve(7, nodes, horizon=99)
    assert first == scenario.resolve(7, nodes, horizon=99)
    assert first.random_crashes is None
    named, *picked = first.crashes
    assert named == Crash("a", 1.0)
    assert len({c.node for c in picked}) == 2 and "a" not in {c.node for c in picked}
    assert all(10 <= c.at <= 20 and c.restart_at == c.at + 5 for c in picked)
    assert any(scenario.resolve(seed, nodes, 99) != first for seed in range(8, 12))
    with pytest.raises(ValueError, match="2 crashes but 1 nodes"):
        scenario.resolve(7, ["a", "b"], horizon=99)
    assert Scenario().resolve(7, nodes, horizon=99) == Scenario()


def test_random_crashes_default_to_while_jobs_arrive():
    # The window defaults to [0, horizon], drawn exactly as the experiments'
    # old --crashes flag did, so recorded results still reproduce.
    resolved = Scenario(random_crashes=RandomCrashes(2)).resolve(3, ["b", "a", "c"], horizon=20)
    rng = Random("crashes:3")
    expected = [(node, rng.uniform(0, 20)) for node in rng.sample(["a", "b", "c"], 2)]
    assert sorted((c.node, c.at) for c in resolved.crashes) == sorted(expected)


def test_partitions_cut_both_ways_but_not_inside():
    partition = Partition(frozenset({"a", "b"}), at=1, heal_at=3)
    rng = Random(0)
    assert is_dropped((), (partition,), rng, 2, "a", "c", "X")
    assert is_dropped((), (partition,), rng, 2, "c", "b", "X")
    assert not is_dropped((), (partition,), rng, 2, "a", "b", "X")
    assert not is_dropped((), (partition,), rng, 3, "a", "c", "X")  # healed
    assert not is_dropped((), (partition,), rng, 0.5, "a", "c", "X")


def simulator():
    ids = ["a", "b", "c"]
    nodes = clusters.gossip(
        [NodeSpec(node_id, [GpuGroup("A100", 1, 80)]) for node_id in ids],
        neighborhoods=topology.full_mesh(ids),
        placement=FirstFit(backfill=True),
        forwarding=RingForwarding(ids),
        interval=0.5,
        max_staleness=3.0,
        retry_interval=0.25,
    )
    return Simulator(nodes, latency=0.02)


def test_simulator_applies_crashes_like_direct_calls():
    def run(apply):
        sim = simulator()
        sim.submit(Job("j1", 1, 40, {"A100": 10.0}), at=1.0, node_id="a")  # runs on a
        sim.submit(Job("j2", 1, 40, {"A100": 30.0}), at=1.0, node_id="b")  # runs on b
        apply(sim)
        return sim, sim.run(until=100, until_jobs_done=True)

    applied, result = run(lambda sim: sim.apply(Scenario((Crash("a", 3, 8), Crash("a", 12, 14)))))

    def direct(sim):
        sim.crash("a", at=3.0, restart_at=8.0)
        sim.crash("a", at=12.0, restart_at=14.0)

    assert result == run(direct)[1]
    # a owned j1, so the job is lost with a's scheduler; j2 keeps the run
    # going past a's second restart.
    assert result.lost_jobs == {"j1"}
    assert applied.nodes["a"].ctx.incarnation == 3


def test_a_partitioned_worker_keeps_running_and_the_job_runs_twice():
    sim = simulator()
    sim.submit(Job("j0", 1, 40, {"A100": 30.0}), at=1.0, node_id="a")  # fills a
    sim.submit(Job("j1", 1, 40, {"A100": 10.0}), at=2.0, node_id="a")  # runs on b
    sim.apply(Scenario(partitions=(Partition(frozenset({"b"}), at=4, heal_at=20),)))
    result = sim.run(until=100, until_jobs_done=True)

    [interrupted] = [o.metric for o in result.observations if isinstance(o.metric, JobInterrupted)]
    assert (interrupted.job_id, interrupted.worker) == ("j1", "b")
    j1 = [o.metric.worker for o in result.observations if isinstance(o.metric, JobStarted)]
    assert j1.count("b") == 1 and j1.count("c") == 1
    # b was cut off, not crashed: its run finished too, so j1 ran twice.
    assert sorted((e.node_id, e.interrupted) for e in result.executions if e.job_id == "j1") == [
        ("b", False),
        ("c", False),
    ]
    completed = [o.metric.job_id for o in result.observations if isinstance(o.metric, JobCompleted)]
    assert sorted(completed) == ["j0", "j1"]
    assert result.dropped["Heartbeat"] > 0


def test_simulator_refuses_unresolved_or_unknown_scenarios():
    with pytest.raises(ValueError, match="resolve"):
        simulator().apply(Scenario(random_crashes=RandomCrashes(1)))
    with pytest.raises(ValueError, match="unknown nodes"):
        simulator().apply(Scenario(losses=(MessageLoss(senders={"zz"}),)))
