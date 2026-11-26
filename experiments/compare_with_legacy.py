"""
Compare the legacy global-gossip simulator with the new full-mesh gossip.

Every seed generates one workload that all designs run, so differences
between designs are paired per seed. Prints mean scheduling delay with 95%
confidence intervals, the paired differences, and mean rejected
reservations. Run from the repository root:

    python -m experiments.compare_with_legacy --seeds 100 --out results.json

This depends on the legacy modules (`simulator`, `distributed`,
`experiment`) and must be updated or removed when they are deleted.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from statistics import NormalDist, mean, stdev

import distributed as legacy_distributed
import simulator as legacy_simulator
from experiment import make_workload
from torchsched import clusters, topology
from torchsched.forwarding.ring import RingForwarding
from torchsched.placement.first_fit import FirstFit
from torchsched.runtime.sim import Simulator
from torchsched.types.metrics import JobStarted, ReservationRejected
from torchsched.types.models import GpuGroup, Job, NodeSpec

DESIGNS = ("legacy", "strict", "backfill")


def legacy_cluster(l40_nodes: int, a100_nodes: int) -> list[legacy_simulator.Node]:
    return [
        legacy_simulator.Node(f"{gpu_type.lower()}-{i + 1}", gpu_type, 2, vram)
        for gpu_type, count, vram in [("L40", l40_nodes, 48), ("A100", a100_nodes, 80)]
        for i in range(count)
    ]


def run_legacy(nodes, seed: int, args) -> tuple[float, int]:
    """
    Mean scheduling delay and rejected reservations, legacy global gossip.
    """
    jobs = make_workload(seed, args.jobs, args.arrival_rate)
    settings = legacy_distributed.Settings(
        policy="global-gossip",
        seed=seed,
        update_interval=args.interval,
        max_staleness=args.max_staleness,
        latency=args.latency,
        retry_interval=args.retry_interval,
    )
    result, traffic = legacy_distributed.simulate_distributed(nodes, jobs, settings)
    if len(result.placements) != len(jobs):
        raise RuntimeError(f"legacy seed {seed}: not every job completed")
    delay = legacy_simulator.summarize(nodes, result).mean_scheduling_delay
    return delay, traffic.failed_reservations


def run_new(nodes, seed: int, args, *, backfill: bool) -> tuple[float, int]:
    """
    Mean scheduling delay and rejected reservations, new full-mesh gossip.
    """
    specs = [NodeSpec(n.name, [GpuGroup(n.gpu_type, n.gpu_count, n.vram_gb)]) for n in nodes]
    ids = [spec.node_id for spec in specs]
    sim = Simulator(
        clusters.gossip(
            specs,
            neighborhoods=topology.full_mesh(ids),
            placement=FirstFit(backfill=backfill),
            forwarding=RingForwarding(ids),
            interval=args.interval,
            max_staleness=args.max_staleness,
            retry_interval=args.retry_interval,
            seed=seed,
        ),
        latency=args.latency,
    )
    jobs = make_workload(seed, args.jobs, args.arrival_rate)
    for index, job in enumerate(jobs):
        sim.submit(
            Job(job.name, job.gpu_count, job.min_vram_gb, job.runtime_by_gpu),
            at=job.arrival,
            node_id=ids[index % len(ids)],
        )
    result = sim.run(until=args.time_limit, until_jobs_done=True)
    starts = {
        o.metric.job_id: o.time
        for o in result.observations
        if isinstance(o.metric, JobStarted)
    }
    if len(starts) != len(jobs):
        raise RuntimeError(f"seed {seed}: not every job started")
    delay = mean(starts[job.name] - job.arrival for job in jobs)
    rejected = sum(isinstance(o.metric, ReservationRejected) for o in result.observations)
    return delay, rejected


def confidence_interval(values: list[float]) -> tuple[float, float, float]:
    """
    Mean and a 95% normal-approximation interval (fine for 30+ samples).
    """
    center = mean(values)
    half = NormalDist().inv_cdf(0.975) * stdev(values) / math.sqrt(len(values))
    return center, center - half, center + half


def report(rows: list[dict]) -> None:
    print(f"{'':12s}{'mean delay':>12s}   {'95% CI':18s} rejected (mean)")
    for design in DESIGNS:
        center, low, high = confidence_interval([row[design] for row in rows])
        rejected = mean(row[f"{design}_rejected"] for row in rows)
        print(f"{design:12s}{center:12.3f}   [{low:.3f}, {high:.3f}]   {rejected:.0f}")

    legacy_mean = mean(row["legacy"] for row in rows)
    print("\npaired differences (same workload per seed):")
    for a, b in [("strict", "legacy"), ("backfill", "legacy"), ("backfill", "strict")]:
        diffs = [row[a] - row[b] for row in rows]
        center, low, high = confidence_interval(diffs)
        slower = sum(diff > 0 for diff in diffs)
        print(
            f"  {a} - {b}: {center:+.3f} s ({center / legacy_mean:+.1%})  "
            f"95% CI [{low:+.3f}, {high:+.3f}]  {a} slower on {slower}/{len(rows)} seeds"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--seeds", type=int, default=100, help="number of seeds")
    parser.add_argument("--first-seed", type=int, default=0)
    parser.add_argument("--jobs", type=int, default=100)
    parser.add_argument("--arrival-rate", type=float, default=5.0)
    parser.add_argument("--l40-nodes", type=int, default=6)
    parser.add_argument("--a100-nodes", type=int, default=6)
    parser.add_argument("--latency", type=float, default=0.02)
    parser.add_argument("--interval", type=float, default=0.5, help="gossip interval")
    parser.add_argument("--max-staleness", type=float, default=3.0)
    parser.add_argument("--retry-interval", type=float, default=0.25)
    parser.add_argument("--time-limit", type=float, default=10_000.0)
    parser.add_argument("--out", help="write per-seed results to this JSON file")
    args = parser.parse_args()
    if args.seeds < 2:
        parser.error("--seeds must be at least 2")
    if args.seeds < 30:
        print("note: fewer than 30 seeds; the normal-approximation CIs are rough\n")

    nodes = legacy_cluster(args.l40_nodes, args.a100_nodes)
    started = time.perf_counter()
    rows = []
    for seed in range(args.first_seed, args.first_seed + args.seeds):
        legacy, legacy_rejected = run_legacy(nodes, seed, args)
        strict, strict_rejected = run_new(nodes, seed, args, backfill=False)
        backfill, backfill_rejected = run_new(nodes, seed, args, backfill=True)
        rows.append(
            dict(
                seed=seed,
                legacy=legacy,
                strict=strict,
                backfill=backfill,
                legacy_rejected=legacy_rejected,
                strict_rejected=strict_rejected,
                backfill_rejected=backfill_rejected,
            )
        )
    print(f"{len(rows)} seeds in {time.perf_counter() - started:.0f} s\n")
    report(rows)
    if args.out:
        with open(args.out, "w") as handle:
            json.dump({"config": vars(args), "rows": rows}, handle, indent=1)


if __name__ == "__main__":
    main()
