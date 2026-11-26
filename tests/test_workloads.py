"""
Tests for the seeded workload generator.
"""

import json
from pathlib import Path

import pytest

from torchsched.workloads import poisson_workload

DATA = Path(__file__).resolve().parent / "data"


def test_matches_the_legacy_generator_exactly():
    """
    Recorded from legacy `experiment.make_workload` in tests/data by
    tests/data/generate_legacy_data.py.
    """
    case = json.loads((DATA / "legacy_workload.json").read_text())["data"]
    workload = poisson_workload(case["seed"], case["jobs_count"], case["arrival_rate"])
    assert len(workload) == len(case["jobs"])
    for submission, expected in zip(workload, case["jobs"]):
        job = submission.job
        assert submission.at == expected["arrival"]
        assert job.job_id == expected["name"]
        assert job.gpu_count == expected["gpu_count"]
        assert job.min_vram_gb == expected["min_vram_gb"]
        assert dict(job.runtime_by_gpu) == expected["runtime_by_gpu"]


def test_arrivals_increase_and_seeds_are_reproducible():
    workload = poisson_workload(1, 50, 2.0)
    times = [s.at for s in workload]
    assert times == sorted(times) and times[0] > 0
    assert poisson_workload(1, 50, 2.0) == workload
    assert poisson_workload(2, 50, 2.0) != workload
    assert poisson_workload(1, 0, 2.0) == []


def test_rejects_invalid_parameters():
    with pytest.raises(ValueError):
        poisson_workload(1, -1, 2.0)
    with pytest.raises(ValueError):
        poisson_workload(1, 5, 0.0)
