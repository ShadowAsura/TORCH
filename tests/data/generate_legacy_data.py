"""
Record legacy simulator outputs that tests compare the new code against.

Run from the repository root while the legacy modules still exist:

    python -m tests.data.generate_legacy_data

Each file records the commit it was generated from. After the legacy code is
deleted, check out that commit to regenerate the data.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import distributed as legacy_distributed
import simulator as legacy_simulator
from experiment import make_workload

HERE = Path(__file__).resolve().parent

# (seed, job count, arrival rate, L40 nodes, A100 nodes); 2 GPUs per node.
CENTRALIZED_CASES = [(42, 100, 5.0, 6, 6), (7, 200, 2.0, 2, 2), (3, 150, 8.0, 3, 1), (11, 80, 0.5, 1, 4)]
# (node count, radius, seed)
RING_CASES = [(12, 1, 42), (7, 2, 3), (2, 1, 0), (1, 1, 5)]
# One workload stored in full, to check the ported generator directly.
WORKLOAD_CASE = (42, 30, 5.0)


def commit() -> str:
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
    return head.stdout.strip()


def legacy_cluster(l40_nodes: int, a100_nodes: int) -> list[legacy_simulator.Node]:
    return [
        legacy_simulator.Node(f"{gpu_type.lower()}-{i + 1}", gpu_type, 2, vram)
        for gpu_type, count, vram in [("L40", l40_nodes, 48), ("A100", a100_nodes, 80)]
        for i in range(count)
    ]


def centralized_timelines() -> list[dict]:
    cases = []
    for seed, count, rate, l40, a100 in CENTRALIZED_CASES:
        result = legacy_simulator.simulate(legacy_cluster(l40, a100), make_workload(seed, count, rate))
        timeline = {p.job.name: [p.start, p.finish, p.node.name] for p in result.placements}
        cases.append(dict(seed=seed, jobs=count, arrival_rate=rate, l40_nodes=l40, a100_nodes=a100, timeline=timeline))
    return cases


def ring_layouts() -> list[dict]:
    return [
        dict(count=count, radius=radius, seed=seed, neighbors=[list(p) for p in legacy_distributed.neighborhoods(count, radius, seed)])
        for count, radius, seed in RING_CASES
    ]


def workload() -> dict:
    seed, count, rate = WORKLOAD_CASE
    jobs = [
        dict(name=j.name, arrival=j.arrival, gpu_count=j.gpu_count, min_vram_gb=j.min_vram_gb, runtime_by_gpu=j.runtime_by_gpu)
        for j in make_workload(seed, count, rate)
    ]
    return dict(seed=seed, jobs_count=count, arrival_rate=rate, jobs=jobs)


def write(name: str, data) -> None:
    payload = {"generated_from": commit(), "data": data}
    (HERE / name).write_text(json.dumps(payload, indent=1) + "\n")
    print(f"wrote {name}")


def main() -> None:
    write("legacy_centralized_timelines.json", centralized_timelines())
    write("legacy_ring_layouts.json", ring_layouts())
    write("legacy_workload.json", workload())


if __name__ == "__main__":
    main()
