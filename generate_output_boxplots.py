#!/usr/bin/env python3
"""Generate boxplots from JSON result files in the outputs directory.

The script supports the result formats currently used in this repository:

1. Standard solver results:
   instance -> "Uncoarsened Greedy" -> metrics
   instance -> "Inflated FullQubo" -> metrics

2. Scale-nested benchmark results:
   instance -> "N5" -> "Uncoarsened ORTools" -> metrics

3. Hyperparameter search results:
   per_instance_trials -> instance -> trial rows with P/radius/alpha/beta.

By default plots are written as SVG files, so the script does not depend on
matplotlib. Use --format png if your Python environment has a working
matplotlib installation.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple
from xml.sax.saxutils import escape


DEFAULT_METRICS = [
    "time_window_violations",
    "total_distance",
    "total_route_duration",
    "computation_time",
    "num_vehicles",
    "capacity_violations",
    "total_waiting_time",
]

METRIC_LABELS = {
    "time_window_violations": "Time Window Violations",
    "total_distance": "Total Travel Distance",
    "total_route_duration": "Total Route Duration",
    "computation_time": "Computation Time (s)",
    "num_vehicles": "Number of Vehicles",
    "capacity_violations": "Capacity Violations",
    "total_waiting_time": "Total Waiting Time",
    "total_service_time": "Total Service Time",
    "total_demand_served": "Total Demand Served",
    "n_coarsened": "Coarsened Node Count",
}

METRIC_ALIASES = {
    "tw_violations": "time_window_violations",
    "elapsed": "computation_time",
}

PARAMETER_ALIASES = {
    "radius": "radiusCoeff",
}

HYPERPARAMETERS = ["P", "radiusCoeff", "alpha", "beta"]

COLORS = [
    "#4C78A8",
    "#F58518",
    "#54A24B",
    "#E45756",
    "#72B7B2",
    "#B279A2",
    "#FF9DA6",
    "#9D755D",
    "#BAB0AC",
]


@dataclass
class SolutionRecord:
    source_file: str
    instance: str
    family: str
    family_group: str
    scale: str
    method: str
    solver: str
    configuration: str
    metrics: Dict[str, float]


@dataclass
class HyperparameterRecord:
    source_file: str
    section: str
    instance: str
    family: str
    family_group: str
    P: Optional[float]
    radiusCoeff: Optional[float]
    alpha: Optional[float]
    beta: Optional[float]
    feasible: Optional[bool]
    metrics: Dict[str, float]


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def normalize_metric_name(name: str) -> str:
    return METRIC_ALIASES.get(name, name)


def normalize_parameter_name(name: str) -> str:
    return PARAMETER_ALIASES.get(name, name)


def slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_") or "plot"


def instance_name(raw_key: str) -> str:
    path = Path(str(raw_key))
    name = path.stem if path.suffix else path.name
    return name or str(raw_key)


def infer_family(instance: str) -> Tuple[str, str]:
    match = re.match(r"^(RC|R|C)(\d)", instance)
    if not match:
        return "Unknown", "Unknown"
    family, series = match.groups()
    return family, f"{family}{series}"


def infer_scale_from_source(source_name: str) -> str:
    match = re.search(r"(\d+)\s*customer", source_name)
    if match:
        return f"N{match.group(1)}"
    return "All"


def parse_solver_key(key: str) -> Tuple[str, str]:
    parts = str(key).split()
    if parts and parts[0] in {"Uncoarsened", "Inflated", "Coarsened"}:
        method = parts[0]
        solver = " ".join(parts[1:]) or "Unknown"
        return method, solver
    return "Unknown", str(key)


def looks_like_metric_record(node: Any) -> bool:
    if not isinstance(node, dict):
        return False

    normalized_numeric_keys = {
        normalize_metric_name(key)
        for key, value in node.items()
        if is_number(value)
    }

    expected_keys = {
        "total_distance",
        "total_route_duration",
        "time_window_violations",
        "capacity_violations",
        "num_vehicles",
        "computation_time",
    }
    return bool(normalized_numeric_keys & expected_keys)


def extract_numeric_metrics(node: Dict[str, Any]) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    for key, value in node.items():
        metric_name = normalize_metric_name(key)
        if is_number(value):
            metrics[metric_name] = float(value)
    return metrics


def extract_solution_records(
    data: Dict[str, Any],
    source_file: str,
    source_name: str,
) -> List[SolutionRecord]:
    records: List[SolutionRecord] = []
    default_scale = infer_scale_from_source(source_name)

    def walk(node: Any, path: List[str]) -> None:
        if looks_like_metric_record(node):
            if len(path) < 2:
                return

            instance = instance_name(path[0])
            family, family_group = infer_family(instance)
            scale = next((item for item in path[1:-1] if re.fullmatch(r"N\d+", item)), default_scale)
            method, solver = parse_solver_key(path[-1])
            configuration = f"{method} {solver}".strip()
            if scale != "All":
                configuration = f"{scale} {configuration}"

            records.append(
                SolutionRecord(
                    source_file=source_file,
                    instance=instance,
                    family=family,
                    family_group=family_group,
                    scale=scale,
                    method=method,
                    solver=solver,
                    configuration=configuration,
                    metrics=extract_numeric_metrics(node),
                )
            )
            return

        if isinstance(node, dict):
            for key, value in node.items():
                walk(value, path + [str(key)])

    walk(data, [])
    return records


def extract_hyperparameter_records(
    data: Dict[str, Any],
    source_file: str,
) -> List[HyperparameterRecord]:
    records: List[HyperparameterRecord] = []

    def build_record(section: str, instance_key: str, row: Dict[str, Any]) -> None:
        instance = instance_name(instance_key)
        family, family_group = infer_family(instance)
        params = {
            normalize_parameter_name(key): float(value)
            for key, value in row.items()
            if normalize_parameter_name(key) in HYPERPARAMETERS and is_number(value)
        }
        metrics = extract_numeric_metrics(
            {
                key: value
                for key, value in row.items()
                if normalize_parameter_name(key) not in HYPERPARAMETERS
            }
        )
        feasible = row.get("feasible")
        if not isinstance(feasible, bool):
            feasible = None

        records.append(
            HyperparameterRecord(
                source_file=source_file,
                section=section,
                instance=instance,
                family=family,
                family_group=family_group,
                P=params.get("P"),
                radiusCoeff=params.get("radiusCoeff"),
                alpha=params.get("alpha"),
                beta=params.get("beta"),
                feasible=feasible,
                metrics=metrics,
            )
        )

    for section in ("per_instance_trials", "per_instance_best"):
        section_data = data.get(section)
        if not isinstance(section_data, dict):
            continue

        for instance_key, value in section_data.items():
            if isinstance(value, list):
                for row in value:
                    if isinstance(row, dict):
                        build_record(section, str(instance_key), row)
            elif isinstance(value, dict):
                build_record(section, str(instance_key), value)

    return records


def load_records(
    input_dir: Path,
    include_files: Optional[Set[str]] = None,
) -> Tuple[List[SolutionRecord], List[HyperparameterRecord]]:
    solution_records: List[SolutionRecord] = []
    hyperparameter_records: List[HyperparameterRecord] = []

    for path in sorted(input_dir.glob("*.json")):
        if include_files is not None and path.name not in include_files:
            continue
        try:
            with path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
        except json.JSONDecodeError as exc:
            print(f"Skipping invalid JSON file {path}: {exc}", file=sys.stderr)
            continue

        if not isinstance(data, dict):
            continue

        source_file = path.name
        source_name = path.name

        if "per_instance_trials" in data or "per_instance_best" in data:
            hyperparameter_records.extend(extract_hyperparameter_records(data, source_file))
        else:
            if "per_instance_results" in data:
                cfg = data.get("config", {})
                if isinstance(cfg, dict) and "customers" in cfg:
                    n = cfg["customers"]
                    source_name = f"{path.stem}_{n}customer{path.suffix}"
                data = data["per_instance_results"]
            solution_records.extend(extract_solution_records(data, source_file, source_name))

    return solution_records, hyperparameter_records


def percentile(sorted_values: Sequence[float], fraction: float) -> float:
    if not sorted_values:
        raise ValueError("percentile requires at least one value")
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = (len(sorted_values) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def box_stats(values: Sequence[float]) -> Dict[str, Any]:
    sorted_values = sorted(values)
    q1 = percentile(sorted_values, 0.25)
    median = percentile(sorted_values, 0.50)
    q3 = percentile(sorted_values, 0.75)
    iqr = q3 - q1
    lower_fence = q1 - 1.5 * iqr
    upper_fence = q3 + 1.5 * iqr
    non_outliers = [value for value in sorted_values if lower_fence <= value <= upper_fence]
    outliers = [value for value in sorted_values if value < lower_fence or value > upper_fence]

    return {
        "q1": q1,
        "median": median,
        "q3": q3,
        "low": min(non_outliers) if non_outliers else sorted_values[0],
        "high": max(non_outliers) if non_outliers else sorted_values[-1],
        "outliers": outliers,
        "count": len(sorted_values),
    }


def nice_ticks(min_value: float, max_value: float, count: int = 6) -> List[float]:
    if min_value == max_value:
        pad = abs(min_value) * 0.1 or 1.0
        min_value -= pad
        max_value += pad

    span = max_value - min_value
    raw_step = span / max(count - 1, 1)
    magnitude = 10 ** math.floor(math.log10(raw_step))
    residual = raw_step / magnitude

    if residual <= 1:
        step = magnitude
    elif residual <= 2:
        step = 2 * magnitude
    elif residual <= 5:
        step = 5 * magnitude
    else:
        step = 10 * magnitude

    start = math.floor(min_value / step) * step
    end = math.ceil(max_value / step) * step
    ticks = []
    current = start
    while current <= end + step * 0.5:
        ticks.append(0.0 if abs(current) < step * 1e-9 else current)
        current += step
    return ticks


def format_tick(value: float) -> str:
    if abs(value) >= 1000:
        return f"{value:,.0f}"
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.2f}".rstrip("0").rstrip(".")


def render_svg_boxplot(
    groups: Sequence[Tuple[str, Sequence[float]]],
    title: str,
    ylabel: str,
    output_path: Path,
) -> None:
    non_empty_groups = [(label, list(values)) for label, values in groups if values]
    if not non_empty_groups:
        return

    stats = [(label, box_stats(values)) for label, values in non_empty_groups]
    all_values = [value for _, values in non_empty_groups for value in values]
    data_min = min(all_values)
    data_max = max(all_values)
    ticks = nice_ticks(data_min, data_max)
    y_min = min(ticks)
    y_max = max(ticks)

    label_max_len = max(len(label) for label, _ in stats)
    bottom_margin = min(230, max(120, int(label_max_len * 4.2)))
    width = max(920, min(2400, 125 * len(stats) + 160))
    height = 620
    left = 96
    right = 34
    top = 76
    bottom = bottom_margin
    plot_width = width - left - right
    plot_height = height - top - bottom

    def x_for(index: int) -> float:
        step = plot_width / len(stats)
        return left + step * (index + 0.5)

    def y_for(value: float) -> float:
        if y_max == y_min:
            return top + plot_height / 2
        return top + (y_max - value) / (y_max - y_min) * plot_height

    step = plot_width / len(stats)
    box_width = max(16, min(50, step * 0.46))
    elements: List[str] = []

    elements.append(f'<rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff"/>')
    elements.append(
        f'<text x="{width / 2:.1f}" y="32" text-anchor="middle" '
        'font-family="Arial, sans-serif" font-size="20" font-weight="700">'
        f"{escape(title)}</text>"
    )
    elements.append(
        f'<text transform="translate(22 {top + plot_height / 2:.1f}) rotate(-90)" '
        'text-anchor="middle" font-family="Arial, sans-serif" font-size="15">'
        f"{escape(ylabel)}</text>"
    )

    for tick in ticks:
        y = y_for(tick)
        elements.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{width - right}" y2="{y:.1f}" '
            'stroke="#d9d9d9" stroke-width="1"/>'
        )
        elements.append(
            f'<text x="{left - 10}" y="{y + 4:.1f}" text-anchor="end" '
            'font-family="Arial, sans-serif" font-size="12" fill="#333333">'
            f"{escape(format_tick(tick))}</text>"
        )

    elements.append(
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" '
        'stroke="#333333" stroke-width="1.2"/>'
    )
    elements.append(
        f'<line x1="{left}" y1="{top + plot_height}" x2="{width - right}" '
        f'y2="{top + plot_height}" stroke="#333333" stroke-width="1.2"/>'
    )

    for index, (label, item) in enumerate(stats):
        x = x_for(index)
        color = COLORS[index % len(COLORS)]
        y_q1 = y_for(item["q1"])
        y_q3 = y_for(item["q3"])
        y_median = y_for(item["median"])
        y_low = y_for(item["low"])
        y_high = y_for(item["high"])
        box_top = min(y_q1, y_q3)
        box_height = max(abs(y_q3 - y_q1), 1.5)
        half_box = box_width / 2

        elements.append(
            f'<line x1="{x:.1f}" y1="{y_high:.1f}" x2="{x:.1f}" y2="{y_low:.1f}" '
            'stroke="#333333" stroke-width="1.2"/>'
        )
        elements.append(
            f'<line x1="{x - half_box * 0.65:.1f}" y1="{y_high:.1f}" '
            f'x2="{x + half_box * 0.65:.1f}" y2="{y_high:.1f}" '
            'stroke="#333333" stroke-width="1.2"/>'
        )
        elements.append(
            f'<line x1="{x - half_box * 0.65:.1f}" y1="{y_low:.1f}" '
            f'x2="{x + half_box * 0.65:.1f}" y2="{y_low:.1f}" '
            'stroke="#333333" stroke-width="1.2"/>'
        )
        elements.append(
            f'<rect x="{x - half_box:.1f}" y="{box_top:.1f}" '
            f'width="{box_width:.1f}" height="{box_height:.1f}" '
            f'fill="{color}" fill-opacity="0.70" stroke="#333333" stroke-width="1.2"/>'
        )
        elements.append(
            f'<line x1="{x - half_box:.1f}" y1="{y_median:.1f}" '
            f'x2="{x + half_box:.1f}" y2="{y_median:.1f}" '
            'stroke="#111111" stroke-width="2"/>'
        )

        for outlier in item["outliers"]:
            y_outlier = y_for(outlier)
            elements.append(
                f'<circle cx="{x:.1f}" cy="{y_outlier:.1f}" r="3" '
                'fill="#ffffff" stroke="#333333" stroke-width="1"/>'
            )

        count_text = f"n={item['count']}"
        elements.append(
            f'<text x="{x:.1f}" y="{top + plot_height + 17}" text-anchor="middle" '
            'font-family="Arial, sans-serif" font-size="10" fill="#555555">'
            f"{escape(count_text)}</text>"
        )
        elements.append(
            f'<text transform="translate({x - 4:.1f} {top + plot_height + 33}) rotate(45)" '
            'text-anchor="start" font-family="Arial, sans-serif" font-size="12" fill="#222222">'
            f"{escape(label)}</text>"
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


def render_matplotlib_boxplot(
    groups: Sequence[Tuple[str, Sequence[float]]],
    title: str,
    ylabel: str,
    output_path: Path,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError(
            "Could not import matplotlib. Use the default SVG output, or fix the "
            "local matplotlib/numpy installation before requesting --format png."
        ) from exc

    non_empty_groups = [(label, list(values)) for label, values in groups if values]
    if not non_empty_groups:
        return

    labels = [label for label, _ in non_empty_groups]
    values = [item for _, item in non_empty_groups]
    width = max(10, min(28, 0.55 * len(labels) + 4))

    plt.figure(figsize=(width, 7))
    boxplot = plt.boxplot(values, labels=labels, patch_artist=True, showfliers=True)
    for index, patch in enumerate(boxplot["boxes"]):
        patch.set_facecolor(COLORS[index % len(COLORS)])
        patch.set_alpha(0.70)
    for median in boxplot["medians"]:
        median.set(color="#111111", linewidth=2)

    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis="y", linestyle="--", alpha=0.45)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


def render_boxplot(
    groups: Sequence[Tuple[str, Sequence[float]]],
    title: str,
    ylabel: str,
    output_path: Path,
    output_format: str,
) -> None:
    if output_format == "svg":
        render_svg_boxplot(groups, title, ylabel, output_path)
    elif output_format == "png":
        render_matplotlib_boxplot(groups, title, ylabel, output_path)
    else:
        raise ValueError(f"Unsupported output format: {output_format}")


def group_solution_values(
    records: Iterable[SolutionRecord],
    metric: str,
    include_source_in_label: bool = False,
) -> List[Tuple[str, List[float]]]:
    grouped: Dict[str, List[float]] = {}
    for record in records:
        if metric not in record.metrics:
            continue
        label = record.configuration
        if include_source_in_label:
            label = f"{Path(record.source_file).stem} | {label}"
        grouped.setdefault(label, []).append(record.metrics[metric])
    return sorted(grouped.items(), key=lambda item: item[0])


def group_hyperparameter_values(
    records: Iterable[HyperparameterRecord],
    metric: str,
    parameter: str,
) -> List[Tuple[str, List[float]]]:
    grouped: Dict[str, List[float]] = {}
    for record in records:
        if record.section != "per_instance_trials":
            continue
        if metric not in record.metrics:
            continue
        parameter_value = getattr(record, parameter)
        if parameter_value is None:
            continue
        label = format_tick(parameter_value)
        grouped.setdefault(label, []).append(record.metrics[metric])

    def sort_key(item: Tuple[str, List[float]]) -> float:
        try:
            return float(item[0])
        except ValueError:
            return math.inf

    return sorted(grouped.items(), key=sort_key)


def selected_metrics(
    requested_metrics: Optional[Sequence[str]],
    records: Sequence[SolutionRecord],
    hyperparameter_records: Sequence[HyperparameterRecord],
) -> List[str]:
    if requested_metrics:
        return [normalize_metric_name(metric) for metric in requested_metrics]

    available = {
        metric
        for record in records
        for metric in record.metrics
    }
    available.update(
        metric
        for record in hyperparameter_records
        for metric in record.metrics
    )
    ordered = [metric for metric in DEFAULT_METRICS if metric in available]
    extras = sorted(available - set(ordered))
    return ordered + extras


def write_solution_csv(records: Sequence[SolutionRecord], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "source_file",
                "instance",
                "family",
                "family_group",
                "scale",
                "method",
                "solver",
                "configuration",
                "metric",
                "value",
            ],
        )
        writer.writeheader()
        for record in records:
            for metric, value in sorted(record.metrics.items()):
                writer.writerow(
                    {
                        "source_file": record.source_file,
                        "instance": record.instance,
                        "family": record.family,
                        "family_group": record.family_group,
                        "scale": record.scale,
                        "method": record.method,
                        "solver": record.solver,
                        "configuration": record.configuration,
                        "metric": metric,
                        "value": value,
                    }
                )


def write_hyperparameter_csv(records: Sequence[HyperparameterRecord], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "source_file",
                "section",
                "instance",
                "family",
                "family_group",
                "P",
                "radiusCoeff",
                "alpha",
                "beta",
                "feasible",
                "metric",
                "value",
            ],
        )
        writer.writeheader()
        for record in records:
            for metric, value in sorted(record.metrics.items()):
                writer.writerow(
                    {
                        "source_file": record.source_file,
                        "section": record.section,
                        "instance": record.instance,
                        "family": record.family,
                        "family_group": record.family_group,
                        "P": record.P,
                        "radiusCoeff": record.radiusCoeff,
                        "alpha": record.alpha,
                        "beta": record.beta,
                        "feasible": record.feasible,
                        "metric": metric,
                        "value": value,
                    }
                )


def generate_solution_plots(
    records: Sequence[SolutionRecord],
    metrics: Sequence[str],
    output_dir: Path,
    output_format: str,
    make_combined: bool,
    make_by_source: bool,
) -> List[Path]:
    written: List[Path] = []

    if make_by_source:
        sources = sorted({record.source_file for record in records})
        for source in sources:
            source_records = [record for record in records if record.source_file == source]
            source_stem = Path(source).stem
            for metric in metrics:
                groups = group_solution_values(source_records, metric)
                if len(groups) < 1:
                    continue
                output_path = (
                    output_dir
                    / "by_source"
                    / source_stem
                    / f"boxplot_{slugify(metric)}.{output_format}"
                )
                title = f"{METRIC_LABELS.get(metric, metric.replace('_', ' ').title())} - {source_stem}"
                render_boxplot(
                    groups,
                    title=title,
                    ylabel=METRIC_LABELS.get(metric, metric.replace("_", " ").title()),
                    output_path=output_path,
                    output_format=output_format,
                )
                written.append(output_path)

    if make_combined:
        for metric in metrics:
            groups = group_solution_values(records, metric, include_source_in_label=True)
            if len(groups) < 1:
                continue
            output_path = output_dir / "combined" / f"boxplot_{slugify(metric)}.{output_format}"
            render_boxplot(
                groups,
                title=f"{METRIC_LABELS.get(metric, metric.replace('_', ' ').title())} - All Output Files",
                ylabel=METRIC_LABELS.get(metric, metric.replace("_", " ").title()),
                output_path=output_path,
                output_format=output_format,
            )
            written.append(output_path)

    return written


def generate_hyperparameter_plots(
    records: Sequence[HyperparameterRecord],
    metrics: Sequence[str],
    output_dir: Path,
    output_format: str,
) -> List[Path]:
    written: List[Path] = []
    trial_records = [record for record in records if record.section == "per_instance_trials"]
    if not trial_records:
        return written

    for metric in metrics:
        for parameter in HYPERPARAMETERS:
            groups = group_hyperparameter_values(trial_records, metric, parameter)
            if len(groups) < 2:
                continue
            output_path = (
                output_dir
                / "hyperparameters"
                / f"boxplot_{slugify(metric)}_by_{slugify(parameter)}.{output_format}"
            )
            metric_label = METRIC_LABELS.get(metric, metric.replace("_", " ").title())
            render_boxplot(
                groups,
                title=f"{metric_label} by {parameter}",
                ylabel=metric_label,
                output_path=output_path,
                output_format=output_format,
            )
            written.append(output_path)

    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate CVRPTW boxplots from JSON files in the outputs directory."
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
        default=Path("plots/output_boxplots"),
        help="Directory where plots and CSV summaries will be written.",
    )
    parser.add_argument(
        "--format",
        choices=["svg", "png"],
        default="svg",
        help="Plot output format. SVG uses only the Python standard library; PNG requires matplotlib.",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=None,
        help="Optional metric names to plot. Aliases tw_violations and elapsed are accepted.",
    )
    parser.add_argument(
        "--no-combined",
        action="store_true",
        help="Do not generate plots combining all non-hyperparameter result files.",
    )
    parser.add_argument(
        "--no-by-source",
        action="store_true",
        help="Do not generate separate plots for each result JSON file.",
    )
    parser.add_argument(
        "--no-hyperparameters",
        action="store_true",
        help="Do not generate hyperparameter sensitivity plots.",
    )
    parser.add_argument(
        "--no-csv",
        action="store_true",
        help="Do not write normalized CSV summaries.",
    )
    parser.add_argument(
        "--files",
        nargs="+",
        default=None,
        metavar="FILENAME",
        help="Specific JSON filenames to load from --input-dir (e.g. results_classical_final.json). "
             "If omitted, all *.json files in --input-dir are loaded.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.input_dir.exists():
        print(f"Input directory not found: {args.input_dir}", file=sys.stderr)
        return 1

    include_files = set(args.files) if args.files else None
    solution_records, hyperparameter_records = load_records(args.input_dir, include_files)
    metrics = selected_metrics(args.metrics, solution_records, hyperparameter_records)

    if not solution_records and not hyperparameter_records:
        print(f"No supported result records found in {args.input_dir}", file=sys.stderr)
        return 1

    if not args.no_csv:
        if solution_records:
            write_solution_csv(solution_records, args.output_dir / "solution_metrics.csv")
        if hyperparameter_records:
            write_hyperparameter_csv(hyperparameter_records, args.output_dir / "hyperparameter_metrics.csv")

    written: List[Path] = []
    if solution_records:
        written.extend(
            generate_solution_plots(
                records=solution_records,
                metrics=metrics,
                output_dir=args.output_dir,
                output_format=args.format,
                make_combined=not args.no_combined,
                make_by_source=not args.no_by_source,
            )
        )

    if hyperparameter_records and not args.no_hyperparameters:
        written.extend(
            generate_hyperparameter_plots(
                records=hyperparameter_records,
                metrics=metrics,
                output_dir=args.output_dir,
                output_format=args.format,
            )
        )

    print(f"Loaded {len(solution_records)} solver result records.")
    print(f"Loaded {len(hyperparameter_records)} hyperparameter records.")
    print(f"Generated {len(written)} plot files in {args.output_dir}.")
    for path in written[:20]:
        print(f"  {path}")
    if len(written) > 20:
        print(f"  ... {len(written) - 20} more")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
