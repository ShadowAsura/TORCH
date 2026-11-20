from torch_net.models import JobRequest
from torch_net.policies import (
    POLICIES,
    WorkerEndpoint,
    is_compatible,
    rank_compatible,
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


