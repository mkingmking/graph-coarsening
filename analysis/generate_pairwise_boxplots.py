#!/usr/bin/env python3
"""Generate focused 2-vs-2 and quantum reference boxplots.

This script is intentionally separate from generate_output_boxplots.py.

Outputs:
1. Pairwise plots:
   Uncoarsened Greedy vs Coarsened Greedy
   Uncoarsened Savings vs Coarsened Savings
   Uncoarsened FQS vs Coarsened FQS
   Uncoarsened APS vs Coarsened APS
   and any other solver that has both methods in the same JSON file.

2. Quantum reference plots:
   Uncoarsened FQS vs Coarsened FQS vs Uncoarsened OR-Tools
   Uncoarsened APS vs Coarsened APS vs Uncoarsened OR-Tools

3. Combined quantum reference plots per scale:
   Uncoarsened FQS, Coarsened FQS, Uncoarsened APS, Coarsened APS,
   and Uncoarsened OR-Tools.

The default output format is SVG, which avoids the current matplotlib/numpy
ABI issue in this environment. PNG can be requested with --format png if the
local matplotlib installation is fixed.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .generate_output_boxplots import (
    DEFAULT_METRICS,
    METRIC_LABELS,
    SolutionRecord,
    display_solver_name,
    load_records,
    normalize_metric_name,
    render_boxplot,
    slugify,
)


PAIRWISE_METHODS = {"Uncoarsened", "Inflated", "Coarsened"}
COARSENED_METHODS = {"Inflated", "Coarsened"}
QUANTUM_SOLVERS = ("FullQubo", "AveragePartitionSolver")
CLASSICAL_SOLVERS = ("Greedy", "Savings")


@dataclass
class PlotDatum:
    comparison_type: str
    context: str
    group_label: str
    source_file: str
    instance: str
    scale: str
    solver: str
    method: str
    metric: str
    value: float


def method_bucket(record: SolutionRecord) -> Optional[str]:
    if record.method == "Uncoarsened":
        return "uncoarsened"
    if record.method in COARSENED_METHODS:
        return "coarsened"
    return None


def method_label(record: SolutionRecord) -> str:
    if record.method in COARSENED_METHODS:
        return "Coars."
    if record.method == "Uncoarsened":
        return "Unc."
    return record.method


def solver_label(solver: str) -> str:
    return display_solver_name(solver)


def metric_label(metric: str) -> str:
    return METRIC_LABELS.get(metric, metric.replace("_", " ").title())


def values_by_instance(records: Iterable[SolutionRecord], metric: str) -> Dict[str, float]:
    values: Dict[str, float] = {}
    for record in records:
        if metric in record.metrics:
            values[record.instance] = record.metrics[metric]
    return values


def selected_metrics(requested_metrics: Optional[Sequence[str]], records: Sequence[SolutionRecord]) -> List[str]:
    if requested_metrics:
        return [normalize_metric_name(metric) for metric in requested_metrics]

    available = {metric for record in records for metric in record.metrics}
    ordered = [metric for metric in DEFAULT_METRICS if metric in available]
    extras = sorted(available - set(ordered))
    return ordered + extras


def source_stem(source_file: str) -> str:
    return Path(source_file).stem


def record_plot_data(
    rows: List[PlotDatum],
    comparison_type: str,
    context: str,
    group_label: str,
    records: Sequence[SolutionRecord],
    metric: str,
    common_instances: Sequence[str],
) -> None:
    by_instance = values_by_instance(records, metric)
    for record in records:
        if record.instance not in common_instances:
            continue
        if metric not in record.metrics:
            continue
        rows.append(
            PlotDatum(
                comparison_type=comparison_type,
                context=context,
                group_label=group_label,
                source_file=record.source_file,
                instance=record.instance,
                scale=record.scale,
                solver=solver_label(record.solver),
                method=method_label(record),
                metric=metric,
                value=by_instance[record.instance],
            )
        )


def generate_pairwise_plots(
    records: Sequence[SolutionRecord],
    metrics: Sequence[str],
    output_dir: Path,
    output_format: str,
) -> Tuple[List[Path], List[PlotDatum]]:
    written: List[Path] = []
    plot_rows: List[PlotDatum] = []

    grouped: Dict[Tuple[str, str, str], Dict[str, List[SolutionRecord]]] = defaultdict(
        lambda: {"uncoarsened": [], "coarsened": []}
    )

    for record in records:
        if record.method not in PAIRWISE_METHODS:
            continue
        bucket = method_bucket(record)
        if bucket is None:
            continue
        key = (record.source_file, record.scale, record.solver)
        grouped[key][bucket].append(record)

    for (source_file, scale, solver), method_records in sorted(grouped.items()):
        uncoarsened_records = method_records["uncoarsened"]
        coarsened_records = method_records["coarsened"]
        if not uncoarsened_records or not coarsened_records:
            continue

        display_solver = solver_label(solver)
        context = f"{source_stem(source_file)} | {scale} | {display_solver}"

        for metric in metrics:
            uncoarsened_values = values_by_instance(uncoarsened_records, metric)
            coarsened_values = values_by_instance(coarsened_records, metric)
            common_instances = sorted(set(uncoarsened_values) & set(coarsened_values))
            if not common_instances:
                continue

            groups = [
                (
                    f"Unc. {display_solver}",
                    [uncoarsened_values[instance] for instance in common_instances],
                ),
                (
                    f"Coars. {display_solver}",
                    [coarsened_values[instance] for instance in common_instances],
                ),
            ]

            output_path = (
                output_dir
                / "pairwise"
                / source_stem(source_file)
                / slugify(scale)
                / slugify(display_solver)
                / f"boxplot_{slugify(metric)}.{output_format}"
            )
            render_boxplot(
                groups=groups,
                title=metric_label(metric),
                ylabel=metric_label(metric),
                output_path=output_path,
                output_format=output_format,
            )
            written.append(output_path)

            record_plot_data(
                plot_rows,
                "pairwise",
                context,
                f"Unc. {display_solver}",
                uncoarsened_records,
                metric,
                common_instances,
            )
            record_plot_data(
                plot_rows,
                "pairwise",
                context,
                f"Coars. {display_solver}",
                coarsened_records,
                metric,
                common_instances,
            )

    return written, plot_rows


def records_for_group(
    records: Sequence[SolutionRecord],
    scale: str,
    solver: str,
    method_bucket_name: str,
) -> List[SolutionRecord]:
    matching: List[SolutionRecord] = []
    for record in records:
        if record.scale != scale or record.solver != solver:
            continue
        if method_bucket(record) != method_bucket_name:
            continue
        matching.append(record)
    return matching


def generate_quantum_reference_plots(
    records: Sequence[SolutionRecord],
    metrics: Sequence[str],
    output_dir: Path,
    output_format: str,
) -> Tuple[List[Path], List[PlotDatum]]:
    written: List[Path] = []
    plot_rows: List[PlotDatum] = []

    scales = sorted(
        {
            record.scale
            for record in records
            if record.scale.startswith("N") and record.solver in {*QUANTUM_SOLVERS, "ORTools"}
        }
    )

    for scale in scales:
        ortools_records = records_for_group(records, scale, "ORTools", "uncoarsened")
        if not ortools_records:
            continue

        for quantum_solver in QUANTUM_SOLVERS:
            uncoarsened_records = records_for_group(records, scale, quantum_solver, "uncoarsened")
            coarsened_records = records_for_group(records, scale, quantum_solver, "coarsened")
            if not uncoarsened_records or not coarsened_records:
                continue

            display_solver = solver_label(quantum_solver)
            context = f"{scale} | {display_solver} vs OR-Tools"

            for metric in metrics:
                uncoarsened_values = values_by_instance(uncoarsened_records, metric)
                coarsened_values = values_by_instance(coarsened_records, metric)
                ortools_values = values_by_instance(ortools_records, metric)
                common_instances = sorted(
                    set(uncoarsened_values) & set(coarsened_values) & set(ortools_values)
                )
                if not common_instances:
                    continue

                groups = [
                    (
                        f"Unc. {display_solver}",
                        [uncoarsened_values[instance] for instance in common_instances],
                    ),
                    (
                        f"Coars. {display_solver}",
                        [coarsened_values[instance] for instance in common_instances],
                    ),
                    (
                        "Unc. OR-Tools",
                        [ortools_values[instance] for instance in common_instances],
                    ),
                ]

                output_path = (
                    output_dir
                    / "quantum_vs_ortools"
                    / slugify(scale)
                    / slugify(display_solver)
                    / f"boxplot_{slugify(metric)}.{output_format}"
                )
                render_boxplot(
                    groups=groups,
                    title=metric_label(metric),
                    ylabel=metric_label(metric),
                    output_path=output_path,
                    output_format=output_format,
                )
                written.append(output_path)

                record_plot_data(
                    plot_rows,
                    "quantum_vs_ortools",
                    context,
                    f"Unc. {display_solver}",
                    uncoarsened_records,
                    metric,
                    common_instances,
                )
                record_plot_data(
                    plot_rows,
                    "quantum_vs_ortools",
                    context,
                    f"Coars. {display_solver}",
                    coarsened_records,
                    metric,
                    common_instances,
                )
                record_plot_data(
                    plot_rows,
                    "quantum_vs_ortools",
                    context,
                    "Unc. OR-Tools",
                    ortools_records,
                    metric,
                    common_instances,
                )

    return written, plot_rows


def generate_combined_quantum_reference_plots(
    records: Sequence[SolutionRecord],
    metrics: Sequence[str],
    output_dir: Path,
    output_format: str,
) -> Tuple[List[Path], List[PlotDatum]]:
    written: List[Path] = []
    plot_rows: List[PlotDatum] = []

    scales = sorted(
        {
            record.scale
            for record in records
            if record.scale.startswith("N") and record.solver in {*QUANTUM_SOLVERS, "ORTools"}
        }
    )

    for scale in scales:
        group_specs = [
            ("Unc. FQS", records_for_group(records, scale, "FullQubo", "uncoarsened")),
            ("Coars. FQS", records_for_group(records, scale, "FullQubo", "coarsened")),
            (
                "Unc. APS",
                records_for_group(records, scale, "AveragePartitionSolver", "uncoarsened"),
            ),
            (
                "Coars. APS",
                records_for_group(records, scale, "AveragePartitionSolver", "coarsened"),
            ),
            ("Unc. OR-Tools", records_for_group(records, scale, "ORTools", "uncoarsened")),
        ]

        if any(not group_records for _, group_records in group_specs):
            continue

        context = f"{scale} | FQS and APS vs OR-Tools"
        for metric in metrics:
            value_maps = [(label, group_records, values_by_instance(group_records, metric)) for label, group_records in group_specs]
            common_instances = sorted(set.intersection(*(set(value_map) for _, _, value_map in value_maps)))
            if not common_instances:
                continue

            groups = [
                (label, [value_map[instance] for instance in common_instances])
                for label, _, value_map in value_maps
            ]

            output_path = (
                output_dir
                / "quantum_vs_ortools"
                / "combined_by_scale"
                / slugify(scale)
                / f"boxplot_{slugify(metric)}.{output_format}"
            )
            render_boxplot(
                groups=groups,
                title=metric_label(metric),
                ylabel=metric_label(metric),
                output_path=output_path,
                output_format=output_format,
            )
            written.append(output_path)

            for label, group_records, _ in value_maps:
                record_plot_data(
                    plot_rows,
                    "quantum_vs_ortools_combined",
                    context,
                    label,
                    group_records,
                    metric,
                    common_instances,
                )

    return written, plot_rows


def generate_classical_ortools_plots(
    records: Sequence[SolutionRecord],
    metrics: Sequence[str],
    output_dir: Path,
    output_format: str,
) -> Tuple[List[Path], List[PlotDatum]]:
    written: List[Path] = []
    plot_rows: List[PlotDatum] = []

    ortools_records = [
        r for r in records
        if r.solver == "ORTools" and r.method == "Uncoarsened" and r.scale == "All"
    ]
    if not ortools_records:
        return written, plot_rows

    for classical_solver in CLASSICAL_SOLVERS:
        uncoarsened_records = [
            r for r in records
            if r.solver == classical_solver and r.method == "Uncoarsened" and r.scale == "All"
        ]
        coarsened_records = [
            r for r in records
            if r.solver == classical_solver and method_bucket(r) == "coarsened" and r.scale == "All"
        ]
        if not uncoarsened_records or not coarsened_records:
            continue

        context = f"Classical | {classical_solver} vs OR-Tools"

        for metric in metrics:
            uncoarsened_values = values_by_instance(uncoarsened_records, metric)
            coarsened_values = values_by_instance(coarsened_records, metric)
            ortools_values = values_by_instance(ortools_records, metric)
            common_instances = sorted(
                set(uncoarsened_values) & set(coarsened_values) & set(ortools_values)
            )
            if not common_instances:
                continue

            groups = [
                (
                    f"Unc. {classical_solver}",
                    [uncoarsened_values[i] for i in common_instances],
                ),
                (
                    f"Coars. {classical_solver}",
                    [coarsened_values[i] for i in common_instances],
                ),
                (
                    "Unc. OR-Tools",
                    [ortools_values[i] for i in common_instances],
                ),
            ]

            output_path = (
                output_dir
                / "classical_vs_ortools"
                / slugify(classical_solver)
                / f"boxplot_{slugify(metric)}.{output_format}"
            )
            render_boxplot(
                groups=groups,
                title=metric_label(metric),
                ylabel=metric_label(metric),
                output_path=output_path,
                output_format=output_format,
            )
            written.append(output_path)

            record_plot_data(
                plot_rows,
                "classical_vs_ortools",
                context,
                f"Unc. {classical_solver}",
                uncoarsened_records,
                metric,
                common_instances,
            )
            record_plot_data(
                plot_rows,
                "classical_vs_ortools",
                context,
                f"Coars. {classical_solver}",
                coarsened_records,
                metric,
                common_instances,
            )
            record_plot_data(
                plot_rows,
                "classical_vs_ortools",
                context,
                "Unc. OR-Tools",
                ortools_records,
                metric,
                common_instances,
            )

    return written, plot_rows


def write_plot_data_csv(rows: Sequence[PlotDatum], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "comparison_type",
                "context",
                "group_label",
                "source_file",
                "instance",
                "scale",
                "solver",
                "method",
                "metric",
                "value",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate pairwise and OR-Tools-reference CVRPTW boxplots from outputs/*.json."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory containing JSON result files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("plots/pairwise_boxplots"),
        help="Directory where focused comparison plots will be written.",
    )
    parser.add_argument(
        "--format",
        choices=["svg", "png"],
        default="svg",
        help="Plot output format. SVG uses the standard-library renderer from generate_output_boxplots.py.",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=None,
        help="Optional metric names to plot, e.g. total_distance computation_time tw_violations.",
    )
    parser.add_argument(
        "--no-pairwise",
        action="store_true",
        help="Skip 2-vs-2 uncoarsened/coarsened plots.",
    )
    parser.add_argument(
        "--no-quantum-reference",
        action="store_true",
        help="Skip FQS/APS vs uncoarsened OR-Tools reference plots.",
    )
    parser.add_argument(
        "--no-combined-quantum-reference",
        action="store_true",
        help="Skip combined FQS/APS/OR-Tools plots.",
    )
    parser.add_argument(
        "--no-classical-reference",
        action="store_true",
        help="Skip Greedy/Savings vs OR-Tools reference plots (requires results_ortools_benchmark.json).",
    )
    parser.add_argument(
        "--no-csv",
        action="store_true",
        help="Do not write the normalized comparison CSV.",
    )
    parser.add_argument(
        "--files",
        nargs="+",
        default=None,
        metavar="FILENAME",
        help="Specific JSON filenames to load from --input-dir. If omitted, all *.json files are loaded.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.input_dir.exists():
        print(f"Input directory not found: {args.input_dir}")
        return 1

    include_files = set(args.files) if args.files else None
    solution_records, _ = load_records(args.input_dir, include_files)
    if not solution_records:
        print(f"No supported solver records found in {args.input_dir}")
        return 1

    metrics = selected_metrics(args.metrics, solution_records)
    written: List[Path] = []
    plot_rows: List[PlotDatum] = []

    if not args.no_pairwise:
        new_paths, new_rows = generate_pairwise_plots(
            records=solution_records,
            metrics=metrics,
            output_dir=args.output_dir,
            output_format=args.format,
        )
        written.extend(new_paths)
        plot_rows.extend(new_rows)

    if not args.no_quantum_reference:
        new_paths, new_rows = generate_quantum_reference_plots(
            records=solution_records,
            metrics=metrics,
            output_dir=args.output_dir,
            output_format=args.format,
        )
        written.extend(new_paths)
        plot_rows.extend(new_rows)

    if not args.no_combined_quantum_reference:
        new_paths, new_rows = generate_combined_quantum_reference_plots(
            records=solution_records,
            metrics=metrics,
            output_dir=args.output_dir,
            output_format=args.format,
        )
        written.extend(new_paths)
        plot_rows.extend(new_rows)

    if not args.no_classical_reference:
        new_paths, new_rows = generate_classical_ortools_plots(
            records=solution_records,
            metrics=metrics,
            output_dir=args.output_dir,
            output_format=args.format,
        )
        written.extend(new_paths)
        plot_rows.extend(new_rows)

    if plot_rows and not args.no_csv:
        write_plot_data_csv(plot_rows, args.output_dir / "comparison_metrics.csv")

    print(f"Loaded {len(solution_records)} solver result records.")
    print(f"Generated {len(written)} focused comparison plot files in {args.output_dir}.")
    for path in written[:20]:
        print(f"  {path}")
    if len(written) > 20:
        print(f"  ... {len(written) - 20} more")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
