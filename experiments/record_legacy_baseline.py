"""
Record legacy simulator results for `experiments.compare_with_legacy`.

Runs the legacy simulator's `--design` policy for each seed and writes the
mean scheduling delay, rejected reservations, and network messages,
together with the setup and the commit, to
experiments/data/legacy_<design>.json. Run from the repository root while
the legacy modules still exist:

    python -m experiments.record_legacy_baseline --design sampling --seeds 100

After the legacy code is deleted, check out the recorded commit to rerun it.
"""

from __future__ import annotations

import argparse
import json
import subprocess

import distributed as legacy_distributed
import simulator as legacy_simulator
from experiment import make_workload
from experiments.compare_with_legacy import SETUP, add_setup_arguments, baseline_path, seeds


def legacy_cluster(l40_nodes: int, a100_nodes: int) -> list[legacy_simulator.Node]:
    return [
        legacy_simulator.Node(f"{gpu_type.lower()}-{i + 1}", gpu_type, 2, vram)
        for gpu_type, count, vram in [("L40", l40_nodes, 48), ("A100", a100_nodes, 80)]
        for i in range(count)
    ]


def run_legacy(nodes, seed: int, args) -> dict:
    jobs = make_workload(seed, args.jobs, args.arrival_rate)
    settings = legacy_distributed.Settings(
        policy=args.design,
        seed=seed,
        radius=args.radius,
        sample_size=args.sample_size,
        update_interval=args.interval,
        max_staleness=args.max_staleness,
        latency=args.latency,
        retry_interval=args.retry_interval,
    )
    result, traffic = legacy_distributed.simulate_distributed(nodes, jobs, settings)
    if len(result.placements) != len(jobs):
        raise RuntimeError(f"legacy seed {seed}: not every job completed")
    return dict(
        seed=seed,
        delay=legacy_simulator.summarize(nodes, result).mean_scheduling_delay,
        rejected=traffic.failed_reservations,
        messages=traffic.summary()["messages"],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    add_setup_arguments(parser)
    args = parser.parse_args()
    nodes = legacy_cluster(args.l40_nodes, args.a100_nodes)
    rows = [run_legacy(nodes, seed, args) for seed in seeds(args)]
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    path = baseline_path(args.design)
    path.parent.mkdir(exist_ok=True)
    payload = {
        "generated_from": head,
        "config": {key: getattr(args, key) for key in SETUP},
        "rows": rows,
    }
    path.write_text(json.dumps(payload, indent=1) + "\n")
    print(f"wrote {len(rows)} seeds to {path}")


if __name__ == "__main__":
    main()
