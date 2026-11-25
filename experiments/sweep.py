"""
Run every scheduling design on paired seeded workloads and sweep protocol parameters.

The default run compares every design with strict FIFO and backfill placement on
the same seeds. `--sweep` varies one parameter at a time while every other
setting stays at its baseline; repeat it for several parameters:

    python -m experiments.sweep --out results/baseline
    python -m experiments.sweep --out results/sweeps --seeds 100 \
        --sweep max-staleness=1,3,6 --sweep interval=0.25,0.5,1 \
        --sweep sample-size=2,3,4 --sweep radius=1,2,3

The output directory gets raw per-seed rows, per-case summary statistics, an
HTML report, and dependency-free SVG charts. Runs are deterministic: the same
arguments and code produce the same metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from collections import defaultdict
from html import escape
from pathlib import Path
from typing import Any

from experiments.charts import write_bar_chart, write_line_chart
from experiments.runner import (
    DESIGNS,
    PLACEMENTS,
    SWEEP_PARAMETERS,
    Setup,
    expand_cases,
    parameter_applies,
    run_cases,
    summarize,
)

METRICS = (
    ("delay", "Mean scheduling delay", 1.0, "s"),
    ("completion_time", "Mean completion time", 1.0, "s"),
    ("makespan", "Mean makespan", 1.0, "s"),
    ("throughput", "Mean throughput", 1.0, "jobs/s"),
    ("utilization", "Mean GPU utilization", 100.0, "%"),
    ("rejected", "Mean rejected reservations", 1.0, ""),
    ("messages", "Mean network messages", 1.0, ""),
    ("traffic_bytes", "Mean network traffic", 0.001, "KB"),
)

ROW_FIELDS = (
    "case_index",
    "design",
    "placement",
    "sweep",
    "value",
    "seed",
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

SUMMARY_FIELDS = (
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
    "seeds",
    "mean_completed",
    "completed_ci95",
    "mean_delay",
    "delay_ci95",
    "mean_p95_delay",
    "p95_delay_ci95",
    "mean_completion_time",
    "completion_time_ci95",
    "mean_p95_completion_time",
    "p95_completion_time_ci95",
    "mean_makespan",
    "makespan_ci95",
    "mean_throughput",
    "throughput_ci95",
    "mean_utilization",
    "utilization_ci95",
    "mean_rejected",
    "rejected_ci95",
    "mean_messages",
    "messages_ci95",
    "mean_traffic_bytes",
    "traffic_bytes_ci95",
)


def parse_sweep(text: str) -> tuple[str, list[float | int]]:
    """
    Parse `name=value,value`; integer-valued parameters stay integers.
    """
    if "=" not in text:
        raise argparse.ArgumentTypeError("sweeps use name=value,value")
    name, raw_values = text.split("=", 1)
    canonical = SWEEP_PARAMETERS.get(name.strip().lower())
    if canonical is None:
        choices = ", ".join(sorted(SWEEP_PARAMETERS))
        raise argparse.ArgumentTypeError(f"unknown sweep {name!r}; choices: {choices}")
    try:
        values = [float(part) for part in raw_values.split(",") if part]
    except ValueError as error:
        raise argparse.ArgumentTypeError("sweep values must be numeric") from error
    if not values:
        raise argparse.ArgumentTypeError("a sweep needs at least one value")
    if canonical in ("radius", "sample_size"):
        if any(value != int(value) for value in values):
            raise argparse.ArgumentTypeError(f"{name} values must be integers")
        values = [int(value) for value in values]
    return canonical, values


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--designs",
        nargs="+",
        choices=DESIGNS,
        default=list(DESIGNS),
        help="designs to compare (default: all)",
    )
    parser.add_argument(
        "--placement",
        choices=("strict", "backfill", "both"),
        default="both",
        help="placement queue discipline (default: both)",
    )
    parser.add_argument("--seeds", type=int, default=30, help="number of seeds")
    parser.add_argument("--first-seed", type=int, default=0)
    parser.add_argument("--jobs", type=int, default=100, help="jobs per seed")
    parser.add_argument("--arrival-rate", type=float, default=5.0)
    parser.add_argument("--l40-nodes", type=int, default=6)
    parser.add_argument("--a100-nodes", type=int, default=6)
    parser.add_argument("--latency", type=float, default=0.02)
    parser.add_argument("--interval", type=float, default=0.5)
    parser.add_argument("--max-staleness", type=float, default=3.0)
    parser.add_argument("--retry-interval", type=float, default=0.25)
    parser.add_argument("--radius", type=int, default=1)
    parser.add_argument("--sample-size", type=int, default=2)
    parser.add_argument("--time-limit", type=float, default=10_000.0)
    parser.add_argument(
        "--sweep",
        action="append",
        type=parse_sweep,
        default=[],
        metavar="NAME=VALUES",
        help="one-factor-at-a-time values, for example interval=0.25,0.5,1",
    )
    parser.add_argument(
        "--processes",
        type=int,
        default=min(4, os.cpu_count() or 1),
        help="parallel worker processes (default: up to 4)",
    )
    parser.add_argument("--out", type=Path, required=True, help="output directory")


def arguments_to_setup(args: argparse.Namespace) -> Setup:
    return Setup(
        jobs=args.jobs,
        arrival_rate=args.arrival_rate,
        l40_nodes=args.l40_nodes,
        a100_nodes=args.a100_nodes,
        latency=args.latency,
        interval=args.interval,
        max_staleness=args.max_staleness,
        retry_interval=args.retry_interval,
        radius=args.radius,
        sample_size=args.sample_size,
        time_limit=args.time_limit,
    )


def combine_sweeps(
    parsed: list[tuple[str, list[float | int]]],
) -> dict[str, list[float | int]]:
    combined: dict[str, list[float | int]] = {}
    for parameter, values in parsed:
        if parameter in combined:
            raise ValueError(f"{parameter.replace('_', '-')} was swept more than once")
        combined[parameter] = values
    return combined


def write_csv(path: Path, rows: list[dict], fields: tuple[str, ...]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def display_label(summary: dict) -> str:
    return f"{summary['design']} / {summary['placement']}"


def metric_value(summary: dict, metric: str, scale: float) -> float:
    return summary[f"mean_{metric}"] * scale


def write_baseline_charts(chart_dir: Path, summaries: list[dict]) -> list[str]:
    baseline = [summary for summary in summaries if summary["sweep"] == "baseline"]
    written = []
    for metric, title, scale, unit in METRICS:
        filename = f"baseline_{metric}.svg"
        entries = [
            (display_label(summary), metric_value(summary, metric, scale))
            for summary in baseline
        ]
        write_bar_chart(
            chart_dir / filename, entries, title=f"{title} by design", unit=unit
        )
        written.append(filename)
    return written


def write_sweep_charts(
    chart_dir: Path, summaries: list[dict], sweeps: dict[str, list[float | int]]
) -> list[str]:
    written = []
    placements = sorted({summary["placement"] for summary in summaries})
    for parameter in sweeps:
        for placement in placements:
            for metric, title, scale, unit in METRICS:
                grouped: dict[str, list[tuple[float, float]]] = defaultdict(list)
                for summary in summaries:
                    if summary["placement"] != placement:
                        continue
                    if summary["sweep"] not in ("baseline", parameter):
                        continue
                    if not parameter_applies(parameter, summary["design"]):
                        continue
                    x = (
                        summary[parameter]
                        if summary["sweep"] == "baseline"
                        else float(summary["value"])
                    )
                    grouped[summary["design"]].append(
                        (x, metric_value(summary, metric, scale))
                    )
                series = [
                    (name, sorted(set(points)))
                    for name, points in sorted(grouped.items())
                    if len({x for x, _ in points}) > 1
                ]
                if not series:
                    continue
                filename = f"sweep_{parameter}_{placement}_{metric}.svg"
                write_line_chart(
                    chart_dir / filename,
                    series,
                    title=(
                        f"{title} vs {parameter.replace('_', '-')} "
                        f"({placement} placement)"
                    ),
                    x_label=parameter.replace("_", "-"),
                    y_label=f"{title} ({unit})" if unit else title,
                )
                written.append(filename)
    return written


def cell(value: Any) -> str:
    if isinstance(value, float):
        return escape(f"{value:.3f}")
    return escape(str(value))


def write_html_report(
    path: Path,
    summaries: list[dict],
    chart_names: list[str],
    config: dict,
) -> None:
    baseline = [summary for summary in summaries if summary["sweep"] == "baseline"]
    columns = (
        ("design", "Design"),
        ("placement", "Placement"),
        ("mean_delay", "Delay (s)"),
        ("mean_completion_time", "Completion (s)"),
        ("mean_makespan", "Makespan (s)"),
        ("mean_throughput", "Throughput"),
        ("mean_utilization", "Utilization"),
        ("mean_rejected", "Rejected"),
        ("mean_messages", "Messages"),
        ("mean_traffic_bytes", "Bytes"),
    )

    def table(rows: list[dict]) -> str:
        header = "".join(f"<th>{escape(label)}</th>" for _, label in columns)
        body = "".join(
            "<tr>"
            + "".join(f"<td>{cell(row[key])}</td>" for key, _ in columns)
            + "</tr>"
            for row in rows
        )
        return f"<table><thead><tr>{header}</tr></thead><tbody>{body}</tbody></table>"

    sweep_rows = [summary for summary in summaries if summary["sweep"] != "baseline"]
    images = "".join(
        f'<figure><img src="charts/{escape(name)}" alt="{escape(name)}">'
        f"<figcaption>{escape(name)}</figcaption></figure>"
        for name in chart_names
    )
    config_items = "".join(
        f"<li><code>{escape(str(key))}</code>: {cell(value)}</li>"
        for key, value in sorted(config.items())
        if key != "sweeps"
    )
    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>TORCH experiment sweep</title>
<style>
body {{ font-family: system-ui, sans-serif; margin: 2rem; color: #111827; }}
table {{ border-collapse: collapse; margin: 1rem 0 2rem; font-size: 0.9rem; }}
th, td {{ border: 1px solid #d1d5db; padding: 0.35rem 0.55rem; text-align: right; }}
th:first-child, td:first-child, th:nth-child(2), td:nth-child(2) {{ text-align: left; }}
figure {{ margin: 1.5rem 0; }} img {{ max-width: 820px; width: 100%; border: 1px solid #e5e7eb; }}
figcaption {{ color: #6b7280; font-size: 0.85rem; margin-top: 0.25rem; }}
</style></head><body>
<h1>TORCH experiment sweep</h1>
<p>Means over seeds; raw rows and confidence intervals are in <code>results.json</code>.</p>
<h2>Configuration</h2><ul>{config_items}</ul>
<h2>Baseline comparison</h2>{table(baseline)}
<h2>Parameter sweep results</h2>{table(sweep_rows) if sweep_rows else "<p>No parameter sweeps requested.</p>"}
<h2>Charts</h2>{images}
</body></html>
"""
    path.write_text(html)


def write_outputs(
    out: Path,
    rows: list[dict],
    summaries: list[dict],
    sweeps: dict[str, list[float | int]],
    config: dict,
) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(
        json.dumps({"config": config, "rows": rows, "summary": summaries}, indent=1)
        + "\n"
    )
    write_csv(out / "rows.csv", rows, ROW_FIELDS)
    write_csv(out / "summary.csv", summaries, SUMMARY_FIELDS)
    chart_dir = out / "charts"
    chart_names = write_baseline_charts(chart_dir, summaries)
    chart_names += write_sweep_charts(chart_dir, summaries, sweeps)
    write_html_report(out / "index.html", summaries, chart_names, config)


def print_baseline(summaries: list[dict]) -> None:
    print(
        f"{'case':34s}{'delay':>8s}{'completion':>12s}{'utilization':>13s}"
        f"{'rejected':>10s}{'messages':>10s}{'KB':>8s}"
    )
    for summary in summaries:
        if summary["sweep"] != "baseline":
            continue
        print(
            f"{display_label(summary):34s}"
            f"{summary['mean_delay']:8.3f}{summary['mean_completion_time']:12.3f}"
            f"{summary['mean_utilization']:12.1%}{summary['mean_rejected']:10.1f}"
            f"{summary['mean_messages']:10.0f}{summary['mean_traffic_bytes'] / 1000:8.0f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    add_arguments(parser)
    args = parser.parse_args()
    if args.seeds < 2:
        parser.error("--seeds must be at least 2")
    if args.seeds < 30:
        print("note: fewer than 30 seeds; confidence intervals are rough")
    try:
        setup = arguments_to_setup(args)
        setup.validate()
        sweeps = combine_sweeps(args.sweep)
        for parameter in sweeps:
            if not any(parameter_applies(parameter, design) for design in args.designs):
                parser.error(
                    f"{parameter.replace('_', '-')} does not apply to any selected design"
                )
        placements = PLACEMENTS if args.placement == "both" else (args.placement,)
        cases = expand_cases(setup, args.designs, placements, sweeps)
    except ValueError as error:
        parser.error(str(error))

    started = time.perf_counter()
    rows = run_cases(
        cases,
        first_seed=args.first_seed,
        seed_count=args.seeds,
        processes=args.processes,
    )
    summaries = summarize(rows)
    elapsed = time.perf_counter() - started
    config = {
        "designs": args.designs,
        "placements": list(placements),
        "seeds": args.seeds,
        "first_seed": args.first_seed,
        "processes": args.processes,
        "jobs": args.jobs,
        "arrival_rate": args.arrival_rate,
        "l40_nodes": args.l40_nodes,
        "a100_nodes": args.a100_nodes,
        "latency": args.latency,
        "interval": args.interval,
        "max_staleness": args.max_staleness,
        "retry_interval": args.retry_interval,
        "radius": args.radius,
        "sample_size": args.sample_size,
        "time_limit": args.time_limit,
        "sweeps": sweeps,
    }
    write_outputs(args.out, rows, summaries, sweeps, config)
    print(
        f"{len(rows)} runs ({len(cases)} cases x {args.seeds} seeds) "
        f"in {elapsed:.1f} s; wrote {args.out}\n"
    )
    print_baseline(summaries)


if __name__ == "__main__":
    main()
