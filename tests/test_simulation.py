import pytest

from distributed import Faults, Settings, simulate_distributed
from simulator import Job, Node, summarize


def test_interrupted_work_counts_toward_utilization_without_completions():
    nodes = [Node("a100", "A100", 1, 80)]
    job = Job("interrupted", 0, 1, 64, {"A100": 10})
    result, traffic = simulate_distributed(
        nodes, [job], Settings(max_rounds=1), Faults(worker="a100", crash_at=2)
    )

    assert result.rejected == [job]
    assert not result.placements
    assert traffic.interrupted_jobs == 1
    assert result.interrupted_gpu_seconds == result.elapsed == 2
    assert summarize(nodes, result).gpu_utilization == 1.0


@pytest.mark.parametrize(
    "overrides",
    [
        {"policy": "unknown"},
        {"max_rounds": 0},
        {"sample_size": 0},
        {"radius": -1},
        {"update_interval": 0},
        {"retry_interval": 0},
        {"latency": float("nan")},
        {"max_staleness": -1},
    ],
)
def test_invalid_distributed_settings_fail_before_running(overrides):
    with pytest.raises(ValueError):
        Settings(**overrides)


@pytest.mark.parametrize(
    "overrides",
    [
        {"crash_at": 2},
        {"worker": "a100", "crash_at": 2, "recover_at": 1},
        {"worker": "a100", "recover_at": float("inf")},
        {"drop_status_until": -1},
    ],
)
def test_invalid_fault_settings_fail_early(overrides):
    with pytest.raises(ValueError):
        Faults(**overrides)


def test_unknown_fault_worker_is_not_silently_ignored():
    nodes = [Node("a100", "A100", 1, 80)]
    with pytest.raises(ValueError, match="unknown worker"):
        simulate_distributed(nodes, [], Settings(), Faults(worker="missing"))


@pytest.mark.parametrize(
    "gpu_count,vram", [(0, 80), (1.5, 80), (1, 0), (1, float("inf"))]
)
def test_invalid_node_capacity(gpu_count, vram):
    with pytest.raises(ValueError):
        Node("a100", "A100", gpu_count, vram)


@pytest.mark.parametrize(
    "arrival,count,vram,runtime",
    [
        (-1, 1, 40, 2),
        (float("nan"), 1, 40, 2),
        (0, 0, 40, 2),
        (0, 1.5, 40, 2),
        (0, 1, -1, 2),
        (0, 1, 40, -1),
        (0, 1, 40, float("inf")),
    ],
)
def test_invalid_workload_values(arrival, count, vram, runtime):
    with pytest.raises(ValueError):
        Job("invalid", arrival, count, vram, {"A100": runtime})
