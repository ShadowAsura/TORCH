"""
Compare a new design against the recorded legacy simulator on the same workloads.

Designs (`--design`):

* `global-gossip`: new gossip on a full mesh vs legacy global gossip.
* `gossip`: new gossip vs legacy gossip, both on a ring with shortcuts
  (`--radius`).
* `periodic`: new periodic dissemination vs legacy periodic, both on a
  ring with shortcuts (`--radius`).
* `sampling`: new Sparrow-style sampling vs legacy sampling, both on a
  ring with shortcuts (`--radius`) and probing `--sample-size - 1` peers.

The new design always uses ring forwarding, with strict FIFO and with
backfill placement.

Every seed generates one workload that all designs run, so differences
between designs are paired per seed. Prints mean scheduling delay with 95%
confidence intervals, the paired differences, mean rejected reservations,
and mean network messages and bytes (bytes for the new code only: legacy
had no message encoding). Run from the repository root:

    python -m experiments.compare_with_legacy --design sampling --seeds 100

Legacy results come from experiments/data/legacy_<design>.json, recorded
by `experiments.record_legacy_baseline` at commit `0abb0e6`, the last one
with the legacy code; no legacy code runs here.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from statistics import NormalDist, mean, stdev

from torchsched import clusters, topology
from torchsched.forwarding.ring import RingForwarding
from torchsched.node import Node
from torchsched.placement.first_fit import FirstFit
from torchsched.runtime.sim import Simulator
from torchsched.types.metrics import JobStarted, ReservationRejected
from torchsched.types.models import GpuGroup, NodeSpec
from torchsched.workloads import Submission, poisson_workload

DATA = Path(__file__).resolve().parent / "data"
DESIGN_CHOICES = ("global-gossip", "gossip", "periodic", "sampling")
COLUMNS = ("legacy", "strict", "backfill")
# Settings that must match the recorded baseline for a fair comparison.
SETUP = (
    "design",
    "jobs",
    "arrival_rate",
    "l40_nodes",
    "a100_nodes",
    "latency",
    "interval",
    "max_staleness",
    "retry_interval",
    "radius",
    "sample_size",
)


def baseline_path(design: str) -> Path:
    return DATA / f"legacy_{design.replace('-', '_')}.json"


def add_setup_arguments(parser: argparse.ArgumentParser) -> None:
    """
    The workload and cluster setup, plus which seeds to run.
    """
    parser.add_argument("--design", choices=DESIGN_CHOICES, default="global-gossip")
    parser.add_argument("--seeds", type=int, default=100, help="number of seeds")
    parser.add_argument("--first-seed", type=int, default=0)
    parser.add_argument("--jobs", type=int, default=100)
    parser.add_argument("--arrival-rate", type=float, default=5.0)
    parser.add_argument("--l40-nodes", type=int, default=6)
    parser.add_argument("--a100-nodes", type=int, default=6)
    parser.add_argument("--latency", type=float, default=0.02)
    parser.add_argument(
        "--interval", type=float, default=0.5, help="gossip or periodic push interval"
    )
    parser.add_argument("--max-staleness", type=float, default=3.0)
    parser.add_argument("--retry-interval", type=float, default=0.25)
    parser.add_argument(
        "--radius", type=int, default=1, help="ring radius (all but global-gossip)"
    )
    parser.add_argument(
        "--sample-size", type=int, default=2, help="nodes per lookup, own included"
    )


def seeds(args) -> range:
    return range(args.first_seed, args.first_seed + args.seeds)


def cluster(l40_nodes: int, a100_nodes: int) -> list[NodeSpec]:
    """
    The legacy experiments' cluster: L40 nodes first, then A100 nodes.
    """
    return [
        NodeSpec(f"{gpu_type.lower()}-{i + 1}", [GpuGroup(gpu_type, 2, vram)])
        for gpu_type, count, vram in [("L40", l40_nodes, 48), ("A100", a100_nodes, 80)]
        for i in range(count)
    ]


def build_nodes(args, seed: int, *, backfill: bool) -> list[Node]:
    """
    The new code's nodes for `args.design`, with ring forwarding.
    """
    specs = cluster(args.l40_nodes, args.a100_nodes)
    ids = [spec.node_id for spec in specs]
    shared = dict(
        placement=FirstFit(backfill=backfill),
        forwarding=RingForwarding(ids),
        retry_interval=args.retry_interval,
        seed=seed,
    )
    if args.design == "global-gossip":
        shared["neighborhoods"] = topology.full_mesh(ids)
    else:
        shared["neighborhoods"] = topology.ring_with_shortcuts(
            ids, radius=args.radius, seed=seed
        )
    if args.design == "sampling":
        nodes = clusters.sampling(specs, sample_size=args.sample_size, **shared)
    else:
        build = clusters.periodic if args.design == "periodic" else clusters.gossip
        nodes = build(
            specs, interval=args.interval, max_staleness=args.max_staleness, **shared
        )
    return nodes


def submit_workload(sim: Simulator, args, seed: int) -> list[Submission]:
    """
    The seed's workload, submitted round robin across nodes in cluster order.
    """
    ids = list(sim.nodes)
    workload = poisson_workload(seed, args.jobs, args.arrival_rate)
    for index, submission in enumerate(workload):
        sim.submit(submission.job, at=submission.at, node_id=ids[index % len(ids)])
    return workload


def run_new(seed: int, args, *, backfill: bool) -> tuple[float, int, int, int]:
    """
    Mean scheduling delay, rejected reservations, network messages, and
    network bytes for the new design, with ring forwarding.
    """
    sim = Simulator(build_nodes(args, seed, backfill=backfill), latency=args.latency)
    workload = submit_workload(sim, args, seed)
    result = sim.run(until=args.time_limit, until_jobs_done=True)
    starts: dict[str, float] = {}
    for o in result.observations:
        if isinstance(o.metric, JobStarted):
            starts.setdefault(o.metric.job_id, o.time)
    if len(starts) != len(workload):
        raise RuntimeError(f"seed {seed}: not every job started")
    delay = mean(starts[s.job.job_id] - s.at for s in workload)
    rejected = sum(isinstance(o.metric, ReservationRejected) for o in result.observations)
    return (
        delay,
        rejected,
        sum(result.messages.values()),
        sum(result.message_bytes.values()),
    )


def load_baseline(args, parser: argparse.ArgumentParser) -> dict[int, dict]:
    """
    Recorded legacy rows by seed, after checking the setup matches.
    """
    path = baseline_path(args.design)
    if not path.exists():
        parser.error(f"no recorded legacy baseline for {args.design} ({path})")
    baseline = json.loads(path.read_text())
    mismatched = [
        f"--{key.replace('_', '-')} {getattr(args, key)} (baseline: {baseline['config'][key]})"
        for key in SETUP
        if getattr(args, key) != baseline["config"][key]
    ]
    if mismatched:
        parser.error(
            "settings differ from the recorded legacy baseline: "
            + ", ".join(mismatched)
        )
    rows = {row["seed"]: row for row in baseline["rows"]}
    missing = [seed for seed in seeds(args) if seed not in rows]
    if missing:
        parser.error(f"no recorded legacy results for seeds {missing[:5]}...")
    return rows


def confidence_interval(values: list[float]) -> tuple[float, float, float]:
    """
    Mean and a 95% normal-approximation interval (fine for 30+ samples).
    """
    center = mean(values)
    half = NormalDist().inv_cdf(0.975) * stdev(values) / math.sqrt(len(values))
    return center, center - half, center + half


def report(rows: list[dict]) -> None:
    print(
        f"{'':12s}{'mean delay':>12s}   {'95% CI':18s} "
        f"{'rejected':>9s} {'messages':>9s} {'KB':>8s}  (means per seed)"
    )
    for design in COLUMNS:
        center, low, high = confidence_interval([row[design] for row in rows])
        rejected = mean(row[f"{design}_rejected"] for row in rows)
        messages = mean(row[f"{design}_messages"] for row in rows)
        sizes = [row[f"{design}_bytes"] for row in rows]
        kilobytes = "—" if None in sizes else f"{mean(sizes) / 1000:.0f}"
        print(
            f"{design:12s}{center:12.3f}   [{low:.3f}, {high:.3f}]   "
            f"{rejected:9.0f} {messages:9.0f} {kilobytes:>8s}"
        )

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
    add_setup_arguments(parser)
    parser.add_argument("--time-limit", type=float, default=10_000.0)
    parser.add_argument("--out", help="write per-seed results to this JSON file")
    args = parser.parse_args()
    if args.seeds < 2:
        parser.error("--seeds must be at least 2")
    if args.seeds < 30:
        print("note: fewer than 30 seeds; the normal-approximation CIs are rough\n")
    legacy = load_baseline(args, parser)

    started = time.perf_counter()
    rows = []
    for seed in seeds(args):
        row: dict = {"seed": seed}
        results = {
            "legacy": (
                legacy[seed]["delay"],
                legacy[seed]["rejected"],
                legacy[seed]["messages"],
                None,
            ),
            "strict": run_new(seed, args, backfill=False),
            "backfill": run_new(seed, args, backfill=True),
        }
        for design, (delay, rejected, messages, size) in results.items():
            row[design] = delay
            row[f"{design}_rejected"] = rejected
            row[f"{design}_messages"] = messages
            row[f"{design}_bytes"] = size
        rows.append(row)
    print(f"{len(rows)} seeds in {time.perf_counter() - started:.0f} s\n")
    report(rows)
    if args.out:
        with open(args.out, "w") as handle:
            json.dump({"config": vars(args), "rows": rows}, handle, indent=1)


if __name__ == "__main__":
    main()
