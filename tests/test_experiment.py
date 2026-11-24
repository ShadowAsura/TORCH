import json
import subprocess
import sys
from pathlib import Path

import pytest

from experiment import make_workload

ROOT = Path(__file__).resolve().parents[1]


def experiment(*args):
    return subprocess.run(
        [sys.executable, "experiment.py", *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


@pytest.mark.parametrize("count,rate", [(-1, 1), (2.5, 1), (3, 0), (3, float("nan"))])
def test_invalid_workload_generator_inputs(count, rate):
    with pytest.raises(ValueError):
        make_workload(42, count, rate)


def test_experiment_cli_is_reproducible_and_reports_all_policies():
    args = ("--policy", "all", "--jobs", "15", "--seed", "7")
    first = experiment(*args)
    second = experiment(*args)
    assert first.returncode == second.returncode == 0
    assert first.stdout == second.stdout
    reports = json.loads(first.stdout)
    assert len(reports) == 5
    assert all(report["completed"] + report["rejected"] == 15 for report in reports)


@pytest.mark.parametrize(
    "args",
    [
        ("--arrival-rate", "nan"),
        ("--policy", "gossip", "--update-interval", "0"),
        ("--policy", "gossip", "--crash-node", "missing"),
    ],
)
def test_experiment_cli_rejects_invalid_configuration(args):
    result = experiment(*args)
    assert result.returncode == 2
    assert "error:" in result.stderr
    assert "Traceback" not in result.stderr
