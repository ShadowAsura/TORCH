"""
Tests for the bookkeeping in `experiments.faults`.
"""

import argparse
import json

import pytest

from experiments.faults import add_fault_arguments, run


def arguments(tmp_path, scenario):
    path = tmp_path / "scenario.json"
    path.write_text(json.dumps(scenario))
    parser = argparse.ArgumentParser()
    add_fault_arguments(parser)
    return parser.parse_args(
        ["--jobs", "40", "--time-limit", "2000", "--design", "gossip", "--scenario", str(path)]
    )


@pytest.mark.parametrize(
    "scenario",
    [
        {"random_crashes": {"count": 2}},
        {"random_crashes": {"count": 2, "downtime": 5}},
        {"losses": [{"probability": 0.1}]},
        {"partitions": [{"nodes": ["l40-1", "a100-1"], "at": 3, "heal_at": 10}]},
        {"crashes": [{"node": "a100-1", "at": 2, "restart_at": 7}]},
    ],
    ids=["crash", "crash-restart", "loss", "partition", "named-crash"],
)
def test_every_job_is_completed_or_lost(tmp_path, scenario):
    args = arguments(tmp_path, scenario)
    clean = run(3, args, faults=False)
    assert (clean["completed"], clean["lost"], clean["wasted"]) == (40, 0, 0.0)
    faulty = run(3, args, faults=True)
    assert faulty["completed"] + faulty["lost"] == 40
    assert set(faulty["jcts"]) == set(faulty["delays"])
    assert all(faulty["jcts"][j] >= faulty["delays"][j] for j in faulty["jcts"])
