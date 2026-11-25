"""
Tests for the experiment sweep runner, aggregation, and report files.
"""

import argparse
import json
import xml.etree.ElementTree as ET
from types import SimpleNamespace

import pytest

from experiments.charts import write_bar_chart, write_line_chart
from experiments.compare_with_legacy import run_new
from experiments.runner import (
    DESIGNS,
    Setup,
    expand_cases,
    run_cases,
    simulate,
    summarize,
)
from experiments.sweep import combine_sweeps, parse_sweep, write_outputs


def tiny_setup():
    return Setup(jobs=8, l40_nodes=1, a100_nodes=1)


def test_sweep_expansion_is_one_factor_at_a_time_and_design_specific():
    cases = expand_cases(
        tiny_setup(),
        placements=["backfill"],
        sweeps={
            "interval": [1.0],
            "max_staleness": [1.0],
            "sample_size": [3],
            "radius": [2],
        },
    )
    assert len(cases) == len(DESIGNS) + 3 + 3 + 1 + 3
    varied = {
        (case.design, case.sweep, case.value)
        for case in cases
        if case.sweep != "baseline"
    }
    assert ("centralized", "interval", 1.0) not in varied
    assert ("global-gossip", "sample_size", 3) not in varied
    assert ("sampling", "sample_size", 3) in varied
    assert ("sampling", "radius", 2) in varied


def test_sweep_parser_normalizes_names_and_integer_parameters():
    assert parse_sweep("max-staleness=1,3.5") == ("max_staleness", [1.0, 3.5])
    assert parse_sweep("sample-size=2,3") == ("sample_size", [2, 3])
    with pytest.raises(argparse.ArgumentTypeError):
        parse_sweep("radius=1.5,2")
    with pytest.raises(argparse.ArgumentTypeError):
        parse_sweep("interval=0.5,0.5")
    with pytest.raises(argparse.ArgumentTypeError):
        parse_sweep("interval=nan,0.5")
    with pytest.raises(ValueError, match="more than once"):
        combine_sweeps([("radius", [1]), ("radius", [2])])


def test_setup_rejects_non_finite_numeric_settings():
    with pytest.raises(ValueError, match="finite"):
        Setup(arrival_rate=float("inf")).validate()


def test_simulation_metrics_are_deterministic_and_internally_consistent():
    case = expand_cases(tiny_setup(), designs=["sampling"], placements=["backfill"])[0]
    first = simulate(case, 3)
    second = simulate(case, 3)
    assert first == second
    assert first["completed"] == tiny_setup().jobs
    assert 0 <= first["delay"] <= first["completion_time"]
    assert 0 <= first["p95_delay"] <= first["p95_completion_time"]
    assert 0 < first["utilization"] <= 1
    assert first["throughput"] == pytest.approx(first["completed"] / first["makespan"])
    assert first["messages"] == sum(first["message_counts"].values())
    assert first["traffic_bytes"] == sum(first["message_bytes"].values())


def test_peer_design_matches_legacy_comparison_runner_for_same_settings():
    args = SimpleNamespace(
        design="global-gossip",
        jobs=8,
        arrival_rate=5.0,
        l40_nodes=1,
        a100_nodes=1,
        latency=0.02,
        interval=0.5,
        max_staleness=3.0,
        retry_interval=0.25,
        radius=1,
        sample_size=2,
        time_limit=10_000.0,
    )
    expected = run_new(2, args, backfill=True)
    case = expand_cases(
        tiny_setup(), designs=["global-gossip"], placements=["backfill"]
    )[0]
    row = simulate(case, 2)
    assert (
        row["delay"],
        row["rejected"],
        row["messages"],
        row["traffic_bytes"],
    ) == expected


def test_run_cases_preserves_case_order_and_summarizes_seed_means():
    cases = expand_cases(
        tiny_setup(), designs=["centralized"], placements=["strict", "backfill"]
    )
    rows = run_cases(cases, first_seed=0, seed_count=2, processes=1)
    assert [row["case_index"] for row in rows] == [0, 0, 1, 1]
    summaries = summarize(rows)
    assert len(summaries) == 2
    for index, summary in enumerate(summaries):
        case_rows = [row for row in rows if row["case_index"] == index]
        assert summary["seeds"] == 2
        assert summary["mean_delay"] == pytest.approx(
            sum(row["delay"] for row in case_rows) / 2
        )
        assert summary["mean_messages"] == pytest.approx(
            sum(row["messages"] for row in case_rows) / 2
        )


def test_charts_are_valid_svg(tmp_path):
    bar = tmp_path / "bar.svg"
    write_bar_chart(bar, [("a", 1.0), ("b", 2.5)], title="bar", unit="s")
    ET.parse(bar)
    line = tmp_path / "line.svg"
    write_line_chart(
        line,
        [("a", [(1.0, 2.0), (2.0, 3.0)]), ("b", [(1.0, 1.0), (2.0, 1.5)])],
        title="line",
        x_label="x",
        y_label="y",
    )
    ET.parse(line)


def test_write_outputs_creates_machine_readable_results_and_report(tmp_path):
    cases = expand_cases(
        tiny_setup(),
        designs=["centralized", "periodic"],
        placements=["backfill"],
        sweeps={"interval": [1.0]},
    )
    rows = run_cases(cases, first_seed=0, seed_count=2, processes=1)
    summaries = summarize(rows)
    sweeps = {"interval": [1.0]}
    config = {"jobs": tiny_setup().jobs, "sweeps": sweeps}
    write_outputs(tmp_path, rows, summaries, sweeps, config)

    result = json.loads((tmp_path / "results.json").read_text())
    assert len(result["rows"]) == 2 * len(cases)
    assert len(result["summary"]) == len(cases)
    assert (tmp_path / "rows.csv").read_text().startswith("case_index,")
    assert "Baseline comparison" in (tmp_path / "index.html").read_text()
    assert (tmp_path / "charts" / "baseline_delay.svg").exists()
    assert (tmp_path / "charts" / "sweep_interval_backfill_delay.svg").exists()
    ET.parse(tmp_path / "charts" / "sweep_interval_backfill_delay.svg")
