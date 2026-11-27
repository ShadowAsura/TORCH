"""
Tests for the bookkeeping in `experiments.faults`.
"""

import argparse

import pytest

from experiments.faults import add_fault_arguments, run


def arguments(*extra):
    parser = argparse.ArgumentParser()
    add_fault_arguments(parser)
    return parser.parse_args(["--jobs", "40", "--time-limit", "2000", *extra])


@pytest.mark.parametrize(
    "faults",
    [("--crashes", "2"), ("--crashes", "2", "--downtime", "5"), ("--loss", "0.1")],
    ids=["crash", "crash-restart", "loss"],
)
def test_every_job_is_completed_or_lost(faults):
    args = arguments("--design", "gossip", *faults)
    clean = run(3, args, faults=False)
    assert (clean["completed"], clean["lost"], clean["wasted"]) == (40, 0, 0.0)
    faulty = run(3, args, faults=True)
    assert faulty["completed"] + faulty["lost"] == 40
    assert set(faulty["jcts"]) == set(faulty["delays"])
    assert all(faulty["jcts"][j] >= faulty["delays"][j] for j in faulty["jcts"])
