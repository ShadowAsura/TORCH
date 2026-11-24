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
