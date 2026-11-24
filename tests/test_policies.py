import pytest

from torch_net.models import JobRequest
from torch_net.policies import (
    POLICIES,
    WorkerEndpoint,
    is_compatible,
    rank_compatible,
    valid_resources,
)


def job():
    return JobRequest(
        job_id="heterogeneous-1",
        gpu_count=1,
        vram_gb=64,
        compatible_gpu_types=["A100"],
        work_units=10,
    )


def test_discovery_counts_and_sampling_seed_are_reproducible():
    workers = [
        WorkerEndpoint(str(index), f"http://worker-{index}:8000") for index in range(6)
    ]
    full = POLICIES["centralized"].select(
        workers, sample_size=2, seed=42, job_id="a", attempt=0
    )
    sampled = POLICIES["sampling"].select(
        workers, sample_size=2, seed=42, job_id="a", attempt=0
    )
    assert full == workers
    assert len(sampled) == 2
    assert len(set(sampled)) == len(sampled)
    assert sampled == POLICIES["sampling"].select(
        workers, sample_size=2, seed=42, job_id="a", attempt=0
    )


def test_heterogeneous_placement_ranks_only_compatible_workers():
    workers = [
        (
            WorkerEndpoint("l40", "http://l40:8000"),
            {
                "gpu_type": "L40",
                "gpu_count": 2,
                "free_gpus": 2,
                "gpu_vram_gb": 48,
                "performance": 1.5,
                "allocated_gpus": 0,
            },
        ),
        (
            WorkerEndpoint("slow-a100", "http://slow:8000"),
            {
                "gpu_type": "A100",
                "gpu_count": 2,
                "free_gpus": 2,
                "gpu_vram_gb": 80,
                "performance": 2.0,
                "allocated_gpus": 0,
            },
        ),
        (
            WorkerEndpoint("fast-a100", "http://fast:8000"),
            {
                "gpu_type": "A100",
                "gpu_count": 2,
                "free_gpus": 2,
                "gpu_vram_gb": 80,
                "performance": 4.0,
                "allocated_gpus": 0,
            },
        ),
    ]
    assert not is_compatible(workers[0][1], job())
    assert [w.name for w, _ in rank_compatible(workers, job())] == [
        "fast-a100",
        "slow-a100",
    ]


@pytest.mark.parametrize(
    "changes",
    [
        {"allocated_gpus": None},
        {"free_gpus": True},
        {"gpu_count": "2"},
        {"free_gpus": 3},
        {"allocated_gpus": 1},
        {"gpu_vram_gb": float("nan")},
        {"gpu_vram_gb": 10**400},
        {"performance": float("inf")},
        {"performance": -2},
        {"gpu_type": []},
    ],
)
def test_malformed_resource_snapshots_are_never_ranked(changes):
    good = {
        "gpu_type": "A100",
        "gpu_count": 2,
        "free_gpus": 2,
        "allocated_gpus": 0,
        "gpu_vram_gb": 80,
        "performance": 2,
    }
    invalid = {**good, **changes}
    assert not valid_resources(invalid)
    assert not is_compatible(invalid, job())
    snapshots = [
        (WorkerEndpoint("bad", "http://bad"), invalid),
        (WorkerEndpoint("good", "http://good"), good),
    ]
    assert [worker.name for worker, _ in rank_compatible(snapshots, job())] == ["good"]


def test_resource_snapshot_requires_all_fields_and_mapping():
    good = {
        "gpu_type": "A100",
        "gpu_count": 2,
        "free_gpus": 2,
        "allocated_gpus": 0,
        "gpu_vram_gb": 80,
        "performance": 2,
    }
    assert valid_resources(good)
    assert not valid_resources([])
    assert not valid_resources(
        {key: value for key, value in good.items() if key != "allocated_gpus"}
    )
