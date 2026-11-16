"""Compare TORCH schedulers on a reproducible synthetic workload."""

import argparse
import json
import math
from dataclasses import asdict
from random import Random

from distributed import POLICIES, Faults, Settings, simulate_distributed
from simulator import Job, Node, simulate, summarize


def make_workload(seed: int, job_count: int, arrival_rate: float) -> list[Job]:
    random = Random(seed)
    arrival = 0.0
    jobs = []
    # GPU count, minimum VRAM per GPU, and runtime range on an L40.
    profiles = [(1, 24, 2, 6), (1, 64, 4, 10), (2, 48, 6, 14)]
    for index in range(job_count):
        arrival += random.expovariate(arrival_rate)
        gpu_count, min_vram, shortest, longest = random.choice(profiles)
        duration = random.uniform(shortest, longest)
        jobs.append(
            Job(
                f"job-{index + 1}",
                arrival,
                gpu_count,
                min_vram,
                {"L40": duration, "A100": duration * 0.75},
            )
        )
    return jobs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--jobs", type=int, default=100)
    parser.add_argument("--arrival-rate", type=float, default=1.0, help="jobs per simulated second")
    parser.add_argument("--l40-nodes", type=int, default=2)
    parser.add_argument("--a100-nodes", type=int, default=2)
    parser.add_argument("--gpus-per-node", type=int, default=2)
    parser.add_argument("--policy", choices=("centralized", *POLICIES, "all"), default="centralized")
    parser.add_argument("--radius", type=int, default=1)
    parser.add_argument("--sample-size", type=int, default=2)
    parser.add_argument("--update-interval", type=float, default=0.5)
    parser.add_argument("--max-staleness", type=float, default=3.0)
    parser.add_argument("--latency", type=float, default=0.02, help="one-way simulated network delay")
    parser.add_argument("--retry-interval", type=float, default=0.25)
    parser.add_argument("--max-rounds", type=int, default=1000, help="per-job routing and retry limit")
    parser.add_argument("--crash-node", help="worker name, for example a100-1")
    parser.add_argument("--crash-at", type=float, default=0.0)
    parser.add_argument("--recover-at", type=float)
    parser.add_argument("--drop-status-until", type=float, default=0.0)
    config = parser.parse_args()
    if config.jobs < 0:
        parser.error("--jobs must be nonnegative")
    if not math.isfinite(config.arrival_rate) or config.arrival_rate <= 0:
        parser.error("--arrival-rate must be finite and positive")
    if min(config.l40_nodes, config.a100_nodes) < 0:
        parser.error("node counts must be nonnegative")
    if config.l40_nodes + config.a100_nodes == 0:
        parser.error("set at least one L40 or A100 node")
    if config.gpus_per_node <= 0:
        parser.error("--gpus-per-node must be positive")
    if min(config.radius, config.sample_size, config.max_rounds) <= 0:
        parser.error("radius, sample size, and max rounds must be positive")
    for name in ("update_interval", "max_staleness", "retry_interval", "latency",
                 "crash_at", "recover_at", "drop_status_until"):
        value = getattr(config, name)
        if value is not None and (not math.isfinite(value) or value < 0):
            parser.error(f"--{name.replace('_', '-')} must be finite and nonnegative")
    if not config.update_interval or not config.retry_interval:
        parser.error("update and retry intervals must be positive")
    if config.crash_at or config.recover_at is not None:
        if not config.crash_node:
            parser.error("--crash-at and --recover-at require --crash-node")
    if config.recover_at is not None and config.recover_at <= config.crash_at:
        parser.error("--recover-at must follow --crash-at")
    if config.policy in ("centralized", "all") and (config.crash_node or config.drop_status_until):
        parser.error("fault injection requires a single distributed policy")

    nodes = [
        Node(f"{gpu_type.lower()}-{index + 1}", gpu_type, config.gpus_per_node, vram)
        for gpu_type, count, vram in [("L40", config.l40_nodes, 48), ("A100", config.a100_nodes, 80)]
        for index in range(count)
    ]
    if config.crash_node is not None and config.crash_node not in {node.name for node in nodes}:
        parser.error(f"unknown worker {config.crash_node}")
    jobs = make_workload(config.seed, config.jobs, config.arrival_rate)
    faults = Faults(config.crash_node, config.crash_at, config.recover_at, config.drop_status_until)
    policies = ("centralized", *POLICIES) if config.policy == "all" else (config.policy,)
    reports = []
    for policy in policies:
        if policy == "centralized":
            result = simulate(nodes, jobs)
            overhead = None
        else:
            settings = Settings(
                policy=policy, seed=config.seed, radius=config.radius, sample_size=config.sample_size,
                update_interval=config.update_interval, max_staleness=config.max_staleness,
                latency=config.latency, retry_interval=config.retry_interval, max_rounds=config.max_rounds,
            )
            result, traffic = simulate_distributed(nodes, jobs, settings, faults)
            overhead = traffic.summary()
        completed = len(result.placements)
        reports.append({
            "scheduler": "centralized_fifo" if policy == "centralized" else policy,
            "config": {**vars(config), "policy": policy},
            "elapsed_seconds": result.elapsed,
            "completed": completed,
            "rejected": len(result.rejected),
            "throughput_jobs_per_second": completed / result.elapsed if result.elapsed else 0.0,
            **asdict(summarize(nodes, result)),
            "overhead": overhead,
        })
    print(json.dumps(reports if config.policy == "all" else reports[0], indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
