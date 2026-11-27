"""
Measure what faults cost a design: the same workloads with and without them.

Faults (any combination):

* `--loss P`: drop every network message with probability P.
* `--status-loss-until T`: drop every `StatusPush` until time T.
* `--crashes N`: crash N distinct nodes (chosen per seed) at random times
  while jobs arrive; with `--downtime D` each restarts D seconds later,
  otherwise it stays down.

Every seed runs one workload twice, without and with the faults, so the
differences are paired. Placement is backfill unless `--strict`. Run from
the repository root:

    python -m experiments.faults --design gossip --crashes 2 --downtime 5
    python -m experiments.faults --design sampling --loss 0.05

Reports, as means per seed: jobs completed and lost (submitted to or owned
by a crashed scheduler); scheduling delay (arrival to first start) and
completion time (arrival to first completion) over jobs that completed in
both runs; interrupted runs; abandoned requests; jobs completed more than
once; GPU-seconds wasted on interrupted or duplicate runs; and messages.
"""

from __future__ import annotations

import argparse
import time
from collections import defaultdict
from random import Random
from statistics import mean

from experiments.compare_with_legacy import (
    add_setup_arguments,
    build_nodes,
    confidence_interval,
    seeds,
    submit_workload,
)
from torchsched.runtime.sim import MessageLoss, Simulator
from torchsched.types.metrics import (
    JobCompleted,
    JobInterrupted,
    JobStarted,
    RequestAbandoned,
)

COLUMNS = ("completed", "lost", "interrupted", "abandoned", "duplicated", "wasted", "messages")


def run(seed: int, args, *, faults: bool) -> dict:
    sim = Simulator(
        build_nodes(args, seed, backfill=not args.strict),
        latency=args.latency,
        losses=_losses(args) if faults else (),
        seed=seed,
    )
    workload = submit_workload(sim, args, seed)
    if faults and args.crashes:
        rng = Random(f"crashes:{seed}")
        horizon = workload[-1].at
        for node_id in rng.sample(sorted(sim.nodes), args.crashes):
            at = rng.uniform(0, horizon)
            restart = at + args.downtime if args.downtime is not None else None
            sim.crash(node_id, at=at, restart_at=restart)
    result = sim.run(until=args.time_limit, until_jobs_done=True)

    arrival = {s.job.job_id: s.at for s in workload}
    started: dict[str, float] = {}
    completed: dict[str, float] = {}
    completions: dict[str, int] = defaultdict(int)
    for o in result.observations:
        if isinstance(o.metric, JobStarted):
            started.setdefault(o.metric.job_id, o.time)
        elif isinstance(o.metric, JobCompleted):
            completed.setdefault(o.metric.job_id, o.time)
            completions[o.metric.job_id] += 1

    # Every run of a job after its first full run is wasted, as is every
    # interrupted run.
    wasted = 0.0
    full_runs: dict[str, int] = defaultdict(int)
    for execution in sorted(result.executions, key=lambda e: e.end):
        gpu_seconds = execution.gpu_count * (execution.end - execution.start)
        if execution.interrupted:
            wasted += gpu_seconds
        else:
            full_runs[execution.job_id] += 1
            if full_runs[execution.job_id] > 1:
                wasted += gpu_seconds
    return {
        "delays": {j: started[j] - arrival[j] for j in completed},
        "jcts": {j: t - arrival[j] for j, t in completed.items()},
        "completed": len(completed),
        "lost": len(result.lost_jobs),
        "interrupted": sum(isinstance(o.metric, JobInterrupted) for o in result.observations),
        "abandoned": sum(isinstance(o.metric, RequestAbandoned) for o in result.observations),
        "duplicated": sum(count > 1 for count in completions.values()),
        "wasted": wasted,
        "messages": sum(result.messages.values()),
    }


def _losses(args) -> list[MessageLoss]:
    losses = []
    if args.loss:
        losses.append(MessageLoss(probability=args.loss))
    if args.status_loss_until:
        losses.append(MessageLoss(end=args.status_loss_until, kinds={"StatusPush"}))
    return losses


def report(pairs: list[tuple[dict, dict]], jobs: int) -> None:
    print(f"{'':10s}" + "".join(f"{c:>12s}" for c in COLUMNS) + "   (means per seed)")
    for label, index in (("no faults", 0), ("faults", 1)):
        cells = [mean(pair[index][c] for pair in pairs) for c in COLUMNS]
        print(f"{label:10s}" + "".join(f"{v:12.1f}" for v in cells))
    print(f"({jobs} jobs per seed; wasted is GPU-seconds)\n")

    print("paired differences, faults - no faults, over jobs completed in both runs:")
    for name, key in (("scheduling delay", "delays"), ("completion time", "jcts")):
        diffs, base = [], []
        for clean, faulty in pairs:
            both = clean[key].keys() & faulty[key].keys()
            if both:
                diffs.append(mean(faulty[key][j] - clean[key][j] for j in both))
                base.append(mean(clean[key][j] for j in both))
        center, low, high = confidence_interval(diffs)
        print(
            f"  {name}: {center:+.3f} s ({center / mean(base):+.1%})  "
            f"95% CI [{low:+.3f}, {high:+.3f}]"
        )


def add_fault_arguments(parser: argparse.ArgumentParser) -> None:
    """
    The setup arguments plus placement and fault options.
    """
    add_setup_arguments(parser)
    parser.add_argument("--strict", action="store_true", help="strict FIFO placement")
    parser.add_argument("--loss", type=float, default=0.0, help="message loss probability")
    parser.add_argument("--status-loss-until", type=float, default=0.0)
    parser.add_argument("--crashes", type=int, default=0, help="nodes to crash per seed")
    parser.add_argument("--downtime", type=float, help="restart crashed nodes after this")
    parser.add_argument("--time-limit", type=float, default=10_000.0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    add_fault_arguments(parser)
    args = parser.parse_args()
    if args.seeds < 2:
        parser.error("--seeds must be at least 2")
    if not 0 <= args.loss < 1:
        parser.error("--loss must be in [0, 1)")
    if args.downtime is not None and args.downtime <= 0:
        parser.error("--downtime must be positive")
    if not (args.loss or args.status_loss_until or args.crashes):
        parser.error("choose at least one fault: --loss, --status-loss-until or --crashes")

    started = time.perf_counter()
    pairs = [(run(s, args, faults=False), run(s, args, faults=True)) for s in seeds(args)]
    print(f"{len(pairs)} seeds in {time.perf_counter() - started:.0f} s\n")
    report(pairs, args.jobs)


if __name__ == "__main__":
    main()
