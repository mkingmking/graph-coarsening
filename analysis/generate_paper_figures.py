#!/usr/bin/env python3
"""Assemble the paper figure assets from regenerated plot outputs."""

from __future__ import annotations

import argparse
import math
import shutil
import subprocess
from pathlib import Path
from statistics import mean
from typing import Dict, List, Sequence, Tuple
from xml.sax.saxutils import escape

from .generate_output_boxplots import (
    COLORS,
    PAPER_AXIS_LABEL_FONT_SIZE,
    PAPER_TICK_FONT_SIZE,
    PAPER_TITLE_FONT_SIZE,
    PAPER_XTICK_FONT_SIZE,
    format_tick,
    load_records,
    nice_ticks,
)


FIGURE_COPIES = [
    (
        "plots/pairwise_boxplots/pairwise/results_simulated_annealing_10customer/n10/aps/boxplot_time_window_violations.png",
        "subfigure_time_window_violations/boxplot_time_window_violations_aps.png",
    ),
    (
        "plots/pairwise_boxplots/pairwise/results_simulated_annealing_10customer/n10/fqs/boxplot_time_window_violations.png",
        "subfigure_time_window_violations/boxplot_time_window_violations_fqs.png",
    ),
    (
        "plots/pairwise_boxplots/pairwise/results_classical_solvers/all/greedy/boxplot_time_window_violations.png",
        "subfigure_time_window_violations/boxplot_time_window_violations_greedy.png",
    ),
    (
        "plots/pairwise_boxplots/pairwise/results_classical_solvers/all/savings/boxplot_time_window_violations.png",
        "subfigure_time_window_violations/boxplot_time_window_violations_savings.png",
    ),
    (
        "plots/pairwise_boxplots/quantum_vs_ortools/combined_by_scale/n5/boxplot_total_distance.png",
        "subfigure_total_distance/boxplot_total_distance_N5.png",
    ),
    (
        "plots/pairwise_boxplots/quantum_vs_ortools/combined_by_scale/n10/boxplot_total_distance.png",
        "subfigure_total_distance/boxplot_total_distance_N10.png",
    ),
    (
        "plots/pairwise_boxplots/classical_vs_ortools/greedy/boxplot_total_distance.png",
        "subfigure_total_distance/boxplot_total_distance_greedy.png",
    ),
    (
        "plots/pairwise_boxplots/classical_vs_ortools/savings/boxplot_total_distance.png",
        "subfigure_total_distance/boxplot_total_distance_savings.png",
    ),
    (
        "plots/output_boxplots/by_source/results_dwave10customer/boxplot_total_distance.png",
        "boxplot_total_distance_dwave.png",
    ),
]


def copy_paper_plots(repo_root: Path, output_dir: Path) -> List[Path]:
    written: List[Path] = []
    for source_rel, dest_rel in FIGURE_COPIES:
        source = repo_root / source_rel
        if not source.exists():
            raise FileNotFoundError(f"Missing generated plot: {source}")
        dest = output_dir / dest_rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)
        written.append(dest)
    return written


def mean_computation_time_data(repo_root: Path) -> List[Tuple[str, float]]:
    records, _ = load_records(repo_root / "outputs")
    rows: List[Tuple[str, float]] = []
    specs = [
        ("N5 Unc FQS", "N5", "FullQubo", "Uncoarsened"),
        ("N5 Co FQS", "N5", "FullQubo", "Inflated"),
        ("N5 Unc APS", "N5", "AveragePartitionSolver", "Uncoarsened"),
        ("N5 Co APS", "N5", "AveragePartitionSolver", "Inflated"),
        ("N10 Unc FQS", "N10", "FullQubo", "Uncoarsened"),
        ("N10 Co FQS", "N10", "FullQubo", "Inflated"),
        ("N10 Unc APS", "N10", "AveragePartitionSolver", "Uncoarsened"),
        ("N10 Co APS", "N10", "AveragePartitionSolver", "Inflated"),
    ]

    for label, scale, solver, method in specs:
        values = [
            record.metrics["computation_time"]
            for record in records
            if record.source_file
            in {
                "results_simulated_annealing_5customer.json",
                "results_simulated_annealing_10customer.json",
            }
            and record.scale == scale
            and record.solver == solver
            and record.method == method
            and "computation_time" in record.metrics
        ]
        if not values:
            raise ValueError(f"No computation-time values for {label}")
        rows.append((label, mean(values)))
    return rows


def write_average_time_svg(rows: Sequence[Tuple[str, float]], output_path: Path) -> None:
    width = 1500
    height = 900
    left = 140
    right = 55
    top = 115
    bottom = 260
    plot_width = width - left - right
    plot_height = height - top - bottom
    values = [value for _, value in rows]
    ticks = nice_ticks(0, max(values))
    y_min = min(ticks)
    y_max = max(ticks)

    def y_for(value: float) -> float:
        if math.isclose(y_max, y_min):
            return top + plot_height / 2
        return top + (y_max - value) / (y_max - y_min) * plot_height

    step = plot_width / len(rows)
    bar_width = min(95, step * 0.58)
    elements: List[str] = [
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff"/>',
        f'<text x="{width / 2:.1f}" y="48" text-anchor="middle" '
        f'font-family="Arial, sans-serif" font-size="{PAPER_TITLE_FONT_SIZE}" '
        f'font-weight="700">Average Computation Time</text>',
        f'<text transform="translate(34 {top + plot_height / 2:.1f}) rotate(-90)" '
        f'text-anchor="middle" font-family="Arial, sans-serif" '
        f'font-size="{PAPER_AXIS_LABEL_FONT_SIZE}">Computation Time (s)</text>',
    ]

    for tick in ticks:
        y = y_for(tick)
        elements.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{width - right}" y2="{y:.1f}" '
            'stroke="#d9d9d9" stroke-width="1"/>'
        )
        elements.append(
            f'<text x="{left - 12}" y="{y + 6:.1f}" text-anchor="end" '
            f'font-family="Arial, sans-serif" font-size="{PAPER_TICK_FONT_SIZE}" '
            f'fill="#333333">{escape(format_tick(tick))}</text>'
        )

    elements.append(
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" '
        'stroke="#333333" stroke-width="1.4"/>'
    )
    elements.append(
        f'<line x1="{left}" y1="{top + plot_height}" x2="{width - right}" '
        f'y2="{top + plot_height}" stroke="#333333" stroke-width="1.4"/>'
    )

    for index, (label, value) in enumerate(rows):
        x = left + step * (index + 0.5)
        y = y_for(value)
        color = COLORS[index % len(COLORS)]
        elements.append(
            f'<rect x="{x - bar_width / 2:.1f}" y="{y:.1f}" '
            f'width="{bar_width:.1f}" height="{top + plot_height - y:.1f}" '
            f'fill="{color}" fill-opacity="0.78" stroke="#333333" stroke-width="1.2"/>'
        )
        elements.append(
            f'<text x="{x:.1f}" y="{y - 10:.1f}" text-anchor="middle" '
            f'font-family="Arial, sans-serif" font-size="{PAPER_TICK_FONT_SIZE}" '
            f'fill="#222222">{escape(format_tick(value))}</text>'
        )
        elements.append(
            f'<text transform="translate({x - 8:.1f} {top + plot_height + 52}) rotate(42)" '
            f'text-anchor="start" font-family="Arial, sans-serif" '
            f'font-size="{PAPER_XTICK_FONT_SIZE}" fill="#222222">{escape(label)}</text>'
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "\n".join(
            [
                '<?xml version="1.0" encoding="UTF-8"?>',
                f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
                f'viewBox="0 0 {width} {height}">',
                *elements,
                "</svg>",
                "",
            ]
        ),
        encoding="utf-8",
    )


def convert_svg_to_png(svg_path: Path, png_path: Path) -> None:
    converter = shutil.which("rsvg-convert")
    if not converter:
        raise RuntimeError("rsvg-convert is required to export PNG paper assets")
    subprocess.run([converter, str(svg_path), "-o", str(png_path)], check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Assemble larger-text paper figures.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("paper/Figures"),
        help="Destination directory for assets referenced by paper/main.tex.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = repo_root / output_dir

    written = copy_paper_plots(repo_root, output_dir)

    avg_svg = output_dir / "avg_computation_time_overall.svg"
    avg_png = output_dir / "avg_computation_time_overall.png"
    write_average_time_svg(mean_computation_time_data(repo_root), avg_svg)
    convert_svg_to_png(avg_svg, avg_png)
    written.extend([avg_svg, avg_png])

    print(f"Wrote {len(written)} paper figure assets to {output_dir}.")
    for path in written:
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
