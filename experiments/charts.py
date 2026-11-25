"""
Small dependency-free SVG charts for experiment reports.
"""

from __future__ import annotations

from collections.abc import Iterable
from html import escape
from pathlib import Path

PALETTE = (
    "#2563eb",
    "#dc2626",
    "#059669",
    "#d97706",
    "#7c3aed",
    "#0891b2",
    "#be185d",
    "#4d7c0f",
    "#9333ea",
    "#ea580c",
)


def number(value: float) -> str:
    """
    Compact chart labels without losing small experiment values.
    """
    if value == 0:
        return "0"
    magnitude = abs(value)
    if magnitude >= 1000:
        return f"{value:,.0f}"
    if magnitude >= 100:
        return f"{value:.0f}"
    if magnitude >= 10:
        return f"{value:.1f}"
    if magnitude >= 1:
        return f"{value:.2f}"
    return f"{value:.3f}"


def _write(path: Path, width: int, height: int, body: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    svg = "\n".join(
        [
            (
                '<svg xmlns="http://www.w3.org/2000/svg" '
                f'width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
            ),
            '<rect width="100%" height="100%" fill="white"/>',
            *body,
            "</svg>",
        ]
    )
    path.write_text(svg + "\n")


def write_bar_chart(
    path: Path,
    entries: list[tuple[str, float]],
    *,
    title: str,
    unit: str = "",
) -> None:
    """
    A horizontal bar chart, easier to read with long design/placement labels.
    """
    if not entries:
        raise ValueError("bar chart needs entries")
    width = 820
    left = 210
    right = 90
    top = 54
    row_height = 32
    bottom = 36
    height = top + row_height * len(entries) + bottom
    plot_width = width - left - right
    maximum = max(value for _, value in entries)
    maximum = maximum if maximum > 0 else 1.0
    body = [
        (
            f'<text x="{width / 2}" y="26" text-anchor="middle" '
            'font-family="sans-serif" font-size="18" font-weight="600">'
            f"{escape(title)}</text>"
        )
    ]
    for tick in range(5):
        fraction = tick / 4
        x = left + plot_width * fraction
        value = maximum * fraction
        body.extend(
            [
                (
                    f'<line x1="{x:.1f}" y1="{top - 8}" x2="{x:.1f}" '
                    f'y2="{height - bottom}" stroke="#e5e7eb"/>'
                ),
                (
                    f'<text x="{x:.1f}" y="{height - 14}" text-anchor="middle" '
                    'font-family="sans-serif" font-size="11" fill="#6b7280">'
                    f"{number(value)}</text>"
                ),
            ]
        )
    for index, (label, value) in enumerate(entries):
        y = top + index * row_height
        center = y + row_height / 2
        length = plot_width * value / maximum
        color = PALETTE[index % len(PALETTE)]
        suffix = f" {unit}" if unit else ""
        body.extend(
            [
                (
                    f'<text x="{left - 10}" y="{center + 4:.1f}" text-anchor="end" '
                    'font-family="sans-serif" font-size="12" fill="#111827">'
                    f"{escape(label)}</text>"
                ),
                (
                    f'<rect x="{left}" y="{y + 6}" width="{length:.1f}" '
                    f'height="{row_height - 12}" rx="3" fill="{color}"/>'
                ),
                (
                    f'<text x="{left + length + 7:.1f}" y="{center + 4:.1f}" '
                    'font-family="sans-serif" font-size="12" fill="#111827">'
                    f"{number(value)}{escape(suffix)}</text>"
                ),
            ]
        )
    _write(path, width, height, body)


def write_line_chart(
    path: Path,
    series: list[tuple[str, list[tuple[float, float]]]],
    *,
    title: str,
    x_label: str,
    y_label: str,
) -> None:
    """
    A small multiple-series line chart with numeric x values.
    """
    points = [point for _, values in series for point in values]
    if not points:
        raise ValueError("line chart needs points")
    width = 820
    height = 460
    left = 78
    right = 178
    top = 50
    bottom = 58
    plot_width = width - left - right
    plot_height = height - top - bottom
    x_values = [x for x, _ in points]
    y_values = [y for _, y in points]
    x_min, x_max = min(x_values), max(x_values)
    y_max = max(y_values)
    y_max = y_max if y_max > 0 else 1.0
    x_span = x_max - x_min or 1.0

    def xscale(x: float) -> float:
        return left + plot_width * (x - x_min) / x_span

    def yscale(y: float) -> float:
        return top + plot_height * (1 - y / y_max)

    body = [
        (
            f'<text x="{width / 2}" y="26" text-anchor="middle" '
            'font-family="sans-serif" font-size="18" font-weight="600">'
            f"{escape(title)}</text>"
        ),
        (
            f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" '
            f'y2="{top + plot_height}" stroke="#111827"/>'
        ),
        (
            f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" '
            'stroke="#111827"/>'
        ),
    ]
    for tick in range(5):
        fraction = tick / 4
        y = top + plot_height * fraction
        value = y_max * (1 - fraction)
        body.extend(
            [
                (
                    f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_width}" '
                    f'y2="{y:.1f}" stroke="#e5e7eb"/>'
                ),
                (
                    f'<text x="{left - 8}" y="{y + 4:.1f}" text-anchor="end" '
                    'font-family="sans-serif" font-size="11" fill="#6b7280">'
                    f"{number(value)}</text>"
                ),
            ]
        )
    ticks = sorted(set(x_values))
    if len(ticks) > 8:
        step = max(1, round(len(ticks) / 7))
        ticks = ticks[::step]
    for value in ticks:
        x = xscale(value)
        body.append(
            f'<text x="{x:.1f}" y="{top + plot_height + 20}" text-anchor="middle" '
            'font-family="sans-serif" font-size="11" fill="#6b7280">'
            f"{number(value)}</text>"
        )
    body.extend(
        [
            (
                f'<text x="{left + plot_width / 2}" y="{height - 9}" text-anchor="middle" '
                'font-family="sans-serif" font-size="12" fill="#374151">'
                f"{escape(x_label)}</text>"
            ),
            (
                f'<text x="17" y="{top + plot_height / 2}" text-anchor="middle" '
                f'transform="rotate(-90 17 {top + plot_height / 2})" '
                'font-family="sans-serif" font-size="12" fill="#374151">'
                f"{escape(y_label)}</text>"
            ),
        ]
    )
    for index, (name, values) in enumerate(series):
        color = PALETTE[index % len(PALETTE)]
        ordered = sorted(values)
        path_data = " ".join(
            f"{'M' if point == 0 else 'L'}{xscale(x):.1f},{yscale(y):.1f}"
            for point, (x, y) in enumerate(ordered)
        )
        body.append(
            f'<path d="{path_data}" fill="none" stroke="{color}" stroke-width="2.2"/>'
        )
        for x, y in ordered:
            body.append(
                f'<circle cx="{xscale(x):.1f}" cy="{yscale(y):.1f}" r="3.2" fill="{color}"/>'
            )
        legend_y = top + index * 22
        body.extend(
            [
                (
                    f'<line x1="{left + plot_width + 18}" y1="{legend_y}" '
                    f'x2="{left + plot_width + 40}" y2="{legend_y}" stroke="{color}" stroke-width="3"/>'
                ),
                (
                    f'<text x="{left + plot_width + 47}" y="{legend_y + 4}" '
                    'font-family="sans-serif" font-size="12" fill="#111827">'
                    f"{escape(name)}</text>"
                ),
            ]
        )
    _write(path, width, height, body)
