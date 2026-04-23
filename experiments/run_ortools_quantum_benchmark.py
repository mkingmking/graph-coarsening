"""
run_ortools_quantum_benchmark.py
--------------------------------
Runs OR-Tools on sub-instances of N=5 and N=10 customers extracted from
each Solomon benchmark file, using the same create_subgraph logic as the
quantum pipeline.

These results serve as the classical benchmark against which quantum
solver solutions are compared.

Usage
-----
    python -m graph_coarsening.run_ortools_quantum_benchmark
    python -m graph_coarsening.run_ortools_quantum_benchmark --data ./solomon_dataset
    python -m graph_coarsening.run_ortools_quantum_benchmark --file C1/C101.csv
    python -m graph_coarsening.run_ortools_quantum_benchmark --output ./outputs/my_benchmark.json

Output JSON format
------------------
{
    "C101": {
        "N5": {
            "Uncoarsened ORTools": { ...metrics..., "computation_time": X, "solver_status": "..." }
        },
        "N10": {
            "Uncoarsened ORTools": { ...metrics..., "computation_time": X, "solver_status": "..." }
        }
    },
    ...
}
"""

import argparse
import json
import logging
import time
from pathlib import Path

from ..graph import Graph
from ..utils import load_graph_from_csv, calculate_route_metrics
from ..ortools_solver import ORToolsVRPTWSolver
from ..runners.main_quantum import create_subgraph

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CUSTOMER_SIZES = [5, 10]
TIME_LIMIT_SECONDS = 10  # Sufficient for tiny sub-instances

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-file processing
# ---------------------------------------------------------------------------

def process_file(csv_path: str) -> dict:
    instance_name = Path(csv_path).stem
    logger.info(f"\n=== {instance_name} ===")

    try:
        full_graph, depot_id, capacity = load_graph_from_csv(csv_path)
    except Exception as exc:
        logger.error(f"Failed to load {csv_path}: {exc}")
        return {}

    instance_results = {}

    for n in CUSTOMER_SIZES:
        key = f"N{n}"
        logger.info(f"  [{key}] Creating sub-instance with {n} customers...")

        subgraph = create_subgraph(full_graph, depot_id, n)
        actual_customers = len(subgraph.nodes) - 1  # exclude depot
        if actual_customers < n:
            logger.warning(
                f"  [{key}] Instance only has {actual_customers} customers, skipping N={n}"
            )
            continue

        logger.info(f"  [{key}] Running OR-Tools (time limit={TIME_LIMIT_SECONDS}s)...")
        t0 = time.perf_counter()
        solver = ORToolsVRPTWSolver(
            subgraph, depot_id, capacity,
            time_limit_seconds=TIME_LIMIT_SECONDS,
        )
        routes, metrics = solver.solve()
        metrics["computation_time"] = time.perf_counter() - t0

        logger.info(
            f"  [{key}] -> vehicles={metrics['num_vehicles']}, "
            f"distance={metrics['total_distance']:.2f}, "
            f"feasible={metrics['is_feasible']}, "
            f"TW violations={metrics['time_window_violations']}, "
            f"status={metrics.get('solver_status', '?')}, "
            f"time={metrics['computation_time']:.3f}s"
        )

        instance_results[key] = {"Uncoarsened ORTools": metrics}

    return instance_results


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary(all_results: dict) -> None:
    W = 80
    print("\n" + "=" * W)
    print(f"{'Instance':<12} {'N':>3}  {'Dist':>9}  {'Veh':>4}  {'TW Viol':>8}  {'Feasible':>8}  {'Time':>8}  {'Status'}")
    print("-" * W)

    for instance in sorted(all_results):
        for key in sorted(all_results[instance]):
            unc = all_results[instance][key].get("Uncoarsened ORTools", {})
            print(
                f"{instance:<12} {key:>3}  "
                f"{unc.get('total_distance', 0.0):>9.2f}  "
                f"{unc.get('num_vehicles', 0):>4}  "
                f"{unc.get('time_window_violations', 0):>8}  "
                f"{'Yes' if unc.get('is_feasible') else 'No':>8}  "
                f"{unc.get('computation_time', 0.0):>7.3f}s  "
                f"{unc.get('solver_status', '?')}"
            )
    print("=" * W)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="OR-Tools benchmark on N=5 and N=10 customer sub-instances."
    )
    parser.add_argument(
        "--data", type=str, default=None,
        help="Directory containing Solomon CSV files (default: ./solomon_dataset)"
    )
    parser.add_argument(
        "--file", type=str, default=None,
        help="Run on a single CSV file instead of the full dataset"
    )
    parser.add_argument(
        "--output", type=str,
        default=str(
            Path(__file__).resolve().parent.parent / "outputs" / "results_ortools_quantum_benchmark.json"
        ),
        help="Path for the output JSON file"
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent.parent

    if args.file:
        csv = Path(args.file)
        if not csv.is_file():
            logger.error(f"File not found: {csv}")
            return
        all_results = {csv.stem: process_file(str(csv))}
    else:
        base_dir = Path(args.data) if args.data else script_dir / "solomon_dataset"
        if not base_dir.exists():
            logger.error(f"Data directory not found: {base_dir}")
            return
        files = sorted(base_dir.rglob("*.csv"))
        if not files:
            logger.warning(f"No CSV files found under {base_dir}")
            return
        logger.info(f"Found {len(files)} instances under {base_dir}")
        all_results = {}
        for f in files:
            all_results[f.stem] = process_file(str(f))

    # Save JSON
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as fh:
        json.dump(all_results, fh, indent=4)
    logger.info(f"\nResults saved to {output_path}")

    print_summary(all_results)


if __name__ == "__main__":
    main()
