"""
Reusable experiment execution and aggregation for seeded scheduling runs.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, replace
from statistics import NormalDist, mean, stdev

from torchsched import clusters, topology
from torchsched.forwarding.ring import RingForwarding
from torchsched.node import Node
from torchsched.placement.first_fit import FirstFit
from torchsched.runtime.sim import Simulator
from torchsched.types.metrics import JobCompleted, JobStarted, ReservationRejected
from torchsched.types.models import GpuGroup, NodeSpec
from torchsched.workloads import PROFILES, poisson_workload

DESIGNS = ("centralized", "global-gossip", "gossip", "periodic", "sampling")
PLACEMENTS = ("strict", "backfill")
PEER_DESIGNS = ("global-gossip", "gossip", "periodic", "sampling")

# CLI spellings are hyphenated; these are the corresponding Setup fields.
SWEEP_PARAMETERS = {
    "interval": "interval",
    "max-staleness": "max_staleness",
    "staleness": "max_staleness",
    "sample-size": "sample_size",
    "radius": "radius",
}
PARAMETER_APPLICABILITY = {
    "interval": frozenset({"global-gossip", "gossip", "periodic"}),
    "max_staleness": frozenset({"global-gossip", "gossip", "periodic"}),
    "sample_size": frozenset({"sampling"}),
    "radius": frozenset({"gossip", "periodic", "sampling"}),
}


@dataclass(frozen=True, slots=True)
class Setup:
    """
    Workload, cluster, network, and protocol settings shared by cases.
    """

    jobs: int = 100
    arrival_rate: float = 5.0
    l40_nodes: int = 6
    a100_nodes: int = 6
    latency: float = 0.02
    interval: float = 0.5
    max_staleness: float = 3.0
    retry_interval: float = 0.25
    radius: int = 1
    sample_size: int = 2
    time_limit: float = 10_000.0

    def validate(self) -> None:
        if self.jobs < 1:
            raise ValueError("jobs must be positive")
        if self.arrival_rate <= 0:
            raise ValueError("arrival_rate must be positive")
        if self.l40_nodes < 0 or self.a100_nodes < 0:
            raise ValueError("node counts must be nonnegative")
        if self.l40_nodes + self.a100_nodes < 1:
            raise ValueError("the cluster needs at least one worker")
        groups = [
            group
            for spec in cluster(self.l40_nodes, self.a100_nodes)
            for group in spec.gpus
        ]
        for gpu_count, min_vram, _, _ in PROFILES:
            if not any(
                group.gpu_type in {"L40", "A100"}
                and gpu_count <= group.count
                and min_vram <= group.vram_gb
                for group in groups
            ):
                raise ValueError("the cluster cannot run every workload profile")
        numeric = (
            "arrival_rate",
            "latency",
            "interval",
            "max_staleness",
            "retry_interval",
            "time_limit",
        )
        if any(not math.isfinite(getattr(self, name)) for name in numeric):
            raise ValueError("experiment settings must be finite")
        if self.latency < 0:
            raise ValueError("latency must be nonnegative")
        for name in ("interval", "retry_interval", "time_limit"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.max_staleness < 0:
            raise ValueError("max_staleness must be nonnegative")
        if self.radius < 1:
            raise ValueError("radius must be positive")
        if self.sample_size < 1:
            raise ValueError("sample_size must be at least 1")


@dataclass(frozen=True, slots=True)
class Case:
    """
    One design/configuration point. Seeds pair workloads across cases.
    """

    design: str
    placement: str
    sweep: str  # "baseline" or the Setup field being varied
    value: float | int | None  # the varied value; None for baseline
    setup: Setup


def cluster(l40_nodes: int, a100_nodes: int) -> list[NodeSpec]:
    """
    The standard experiment cluster: L40 nodes first, then A100 nodes.
    """
    return [
        NodeSpec(f"{gpu_type.lower()}-{i + 1}", [GpuGroup(gpu_type, 2, vram)])
        for gpu_type, count, vram in [
            ("L40", l40_nodes, 48),
            ("A100", a100_nodes, 80),
        ]
        for i in range(count)
    ]


def build_nodes(case: Case, seed: int) -> list[Node]:
    """
    Build a case's nodes, using ring forwarding for peer-to-peer designs.
    """
    setup = case.setup
    specs = cluster(setup.l40_nodes, setup.a100_nodes)
    placement = FirstFit(backfill=case.placement == "backfill")
    if case.design == "centralized":
        return clusters.centralized(specs, placement=placement, seed=seed)

    ids = [spec.node_id for spec in specs]
    neighborhoods = (
        topology.full_mesh(ids)
        if case.design == "global-gossip"
        else topology.ring_with_shortcuts(ids, radius=setup.radius, seed=seed)
    )
    shared = {
        "neighborhoods": neighborhoods,
        "placement": placement,
        "forwarding": RingForwarding(ids),
        "retry_interval": setup.retry_interval,
        "seed": seed,
    }
    if case.design == "sampling":
        return clusters.sampling(specs, sample_size=setup.sample_size, **shared)
    if case.design in ("global-gossip", "gossip"):
        return clusters.gossip(
            specs,
            interval=setup.interval,
            max_staleness=setup.max_staleness,
            **shared,
        )
    if case.design == "periodic":
        return clusters.periodic(
            specs,
            interval=setup.interval,
            max_staleness=setup.max_staleness,
            **shared,
        )
    raise ValueError(f"unknown design {case.design!r}")


def percentile(values: Sequence[float], percent: float) -> float:
    """
    Linearly interpolated percentile, with no input mutation.
    """
    if not values:
        raise ValueError("percentile needs at least one value")
    ordered = sorted(values)
    position = (len(ordered) - 1) * percent / 100
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def simulate(case: Case, seed: int) -> dict:
    """
    Run one seeded workload once and return flat metrics plus traffic by type.
    """
    case.setup.validate()
    sim = Simulator(build_nodes(case, seed), latency=case.setup.latency, seed=seed)
    workload = poisson_workload(seed, case.setup.jobs, case.setup.arrival_rate)
    ids = list(sim.nodes)
    submission_nodes = (
        ["coordinator"] * len(workload)
        if case.design == "centralized"
        else [ids[index % len(ids)] for index in range(len(workload))]
    )
    arrivals: dict[str, float] = {}
    for submission, node_id in zip(workload, submission_nodes):
        arrivals[submission.job.job_id] = submission.at
        sim.submit(submission.job, at=submission.at, node_id=node_id)
    result = sim.run(until=case.setup.time_limit, until_jobs_done=True)

    started: dict[str, float] = {}
    completed: dict[str, float] = {}
    rejected = 0
    for observation in result.observations:
        metric = observation.metric
        if isinstance(metric, JobStarted):
            started.setdefault(metric.job_id, observation.time)
        elif isinstance(metric, JobCompleted):
            completed.setdefault(metric.job_id, observation.time)
        elif isinstance(metric, ReservationRejected):
            rejected += 1

    missing = sorted(set(arrivals) - set(completed))
    if missing:
        raise RuntimeError(
            f"{case.design}/{case.placement} seed {seed}: "
            f"{len(missing)} jobs did not complete, first {missing[:3]}"
        )

    delays = [started[job_id] - arrivals[job_id] for job_id in arrivals]
    completion_times = [completed[job_id] - arrivals[job_id] for job_id in arrivals]
    first_arrival = min(arrivals.values())
    makespan = result.end_time - first_arrival
    busy_gpu_seconds = sum(
        execution.gpu_count * (execution.end - execution.start)
        for execution in result.executions
    )
    gpu_capacity = sum(
        spec.gpu_count for spec in cluster(case.setup.l40_nodes, case.setup.a100_nodes)
    )
    utilization = busy_gpu_seconds / (gpu_capacity * makespan)
    row = {
        "design": case.design,
        "placement": case.placement,
        "sweep": case.sweep,
        "value": case.value,
        "seed": seed,
        **asdict(case.setup),
        "completed": len(completed),
        "delay": mean(delays),
        "p95_delay": percentile(delays, 95),
        "completion_time": mean(completion_times),
        "p95_completion_time": percentile(completion_times, 95),
        "makespan": makespan,
        "throughput": len(completed) / makespan,
        "utilization": utilization,
        "rejected": rejected,
        "messages": sum(result.messages.values()),
        "traffic_bytes": sum(result.message_bytes.values()),
        "message_counts": dict(result.messages),
        "message_bytes": dict(result.message_bytes),
    }
    return row


def parameter_applies(parameter: str, design: str) -> bool:
    return design in PARAMETER_APPLICABILITY[parameter]


def expand_cases(
    setup: Setup,
    designs: Iterable[str] = DESIGNS,
    placements: Iterable[str] = PLACEMENTS,
    sweeps: Mapping[str, Sequence[float | int]] | None = None,
) -> list[Case]:
    """
    Baseline cases plus one-factor-at-a-time variations.

    Every sweep value replaces only its own parameter while all others stay
    at the baseline. The shared baseline point is emitted once and reused by
    every sweep chart. Inapplicable parameters are skipped for a design.
    """
    setup.validate()
    cases: list[Case] = []
    sweeps = sweeps or {}
    for parameter in sweeps:
        if parameter not in PARAMETER_APPLICABILITY:
            raise ValueError(f"unknown sweep parameter {parameter!r}")
    for design in designs:
        if design not in DESIGNS:
            raise ValueError(f"unknown design {design!r}")
        for placement in placements:
            if placement not in PLACEMENTS:
                raise ValueError(f"unknown placement {placement!r}")
            cases.append(Case(design, placement, "baseline", None, setup))
            for parameter, values in sweeps.items():
                if not parameter_applies(parameter, design):
                    continue
                for value in values:
                    if value == getattr(setup, parameter):
                        continue
                    varied = replace(setup, **{parameter: value})
                    varied.validate()
                    cases.append(Case(design, placement, parameter, value, varied))
    return cases


def _run_task(task: tuple[int, Case, int]) -> tuple[int, dict]:
    case_index, case, seed = task
    return case_index, simulate(case, seed)


def run_cases(
    cases: Sequence[Case], *, first_seed: int, seed_count: int, processes: int = 1
) -> list[dict]:
    """
    Run every case on every seed, preserving deterministic result order.
    """
    if seed_count < 1:
        raise ValueError("seed_count must be positive")
    tasks = [
        (case_index, case, seed)
        for case_index, case in enumerate(cases)
        for seed in range(first_seed, first_seed + seed_count)
    ]
    if processes < 1:
        raise ValueError("processes must be positive")
    if processes == 1 or len(tasks) == 1:
        pairs = [_run_task(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=processes) as executor:
            pairs = list(executor.map(_run_task, tasks, chunksize=1))
    rows = []
    for case_index, row in pairs:
        row["case_index"] = case_index
        rows.append(row)
    return rows


def confidence_interval(values: Sequence[float]) -> tuple[float, float]:
    """
    Mean and 95% normal-approximation half-width.
    """
    if not values:
        raise ValueError("confidence interval needs values")
    center = mean(values)
    if len(values) == 1:
        return center, 0.0
    return center, NormalDist().inv_cdf(0.975) * stdev(values) / len(values) ** 0.5


def summarize(rows: Sequence[dict]) -> list[dict]:
    """
    Aggregate rows over seeds within each case.
    """
    grouped: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["case_index"]].append(row)

    metrics = (
        "completed",
        "delay",
        "p95_delay",
        "completion_time",
        "p95_completion_time",
        "makespan",
        "throughput",
        "utilization",
        "rejected",
        "messages",
        "traffic_bytes",
    )
    summaries = []
    for case_index in sorted(grouped):
        rows_for_case = grouped[case_index]
        first = rows_for_case[0]
        summary = {
            key: first[key]
            for key in (
                "case_index",
                "design",
                "placement",
                "sweep",
                "value",
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
                "time_limit",
            )
        }
        summary["seeds"] = len(rows_for_case)
        for metric in metrics:
            values = [row[metric] for row in rows_for_case]
            center, half = confidence_interval(values)
            summary[f"mean_{metric}"] = center
            summary[f"{metric}_ci95"] = half
        types = sorted(
            {kind for row in rows_for_case for kind in row["message_counts"]}
        )
        summary["mean_message_counts"] = {
            kind: mean(row["message_counts"].get(kind, 0) for row in rows_for_case)
            for kind in types
        }
        summary["mean_message_bytes"] = {
            kind: mean(row["message_bytes"].get(kind, 0) for row in rows_for_case)
            for kind in types
        }
        summaries.append(summary)
    return summaries
