"""
run_ortools.py
--------------
Runs the OR-Tools CVRPTW solver on all Solomon benchmark instances,
both uncoarsened (original graph) and coarsened+inflated, and saves
the results to a JSON file.

Stopping criteria
-----------------
By default no stopping criterion is imposed — OR-Tools runs until it
proves optimality or exhausts its search. This is required to match
SOTA / best-known Solomon solutions.

Use --time-limit (seconds) to cap each instance if needed (e.g. for
quick experiments). Use --solution-limit N to stop after N improving
solutions (useful for timing experiments; OR-Tools exits immediately
so perf_counter gives the true elapsed time).

Usage
-----
    python -m graph_coarsening.run_ortools                     # no limit (to optimality)
    python -m graph_coarsening.run_ortools --time-limit 300    # 5-min cap per instance
    python -m graph_coarsening.run_ortools --solution-limit 1  # first feasible only
    python -m graph_coarsening.run_ortools --file C1/C101.csv
    python -m graph_coarsening.run_ortools --output my_results.json

Output JSON format (mirrors results_classical.json)
----------------------------------------------------
{
    "C101": {
        "Uncoarsened ORTools": { ...metrics..., "computation_time": X, "solver_status": "..." },
        "Inflated ORTools":    { ...metrics..., "computation_time": X, "solver_status": "..." }
    },
    ...
}
"""

import argparse
import json
import logging
import os
import time
from pathlib import Path

from .graph import Graph
from .utils import load_graph_from_csv, calculate_route_metrics
from .coarsener import SpatioTemporalGraphCoarsener
from .ortools_solver import ORToolsVRPTWSolver

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Coarsening hyperparameters (same as main.py)
COARSENER_PARAMS = dict(alpha=0.8, beta=0.4, P=0.5, radiusCoeff=2.0)


# ---------------------------------------------------------------------------
# Per-file processing
# ---------------------------------------------------------------------------

def process_file(
    csv_path: str,
    solution_limit: int = None,
    time_limit: int = None,
) -> dict:
    """
    solution_limit : stop after N improving solutions (None = no limit).
    time_limit     : hard cap in seconds (None = no limit).
    When both are None, OR-Tools runs to proven optimality.
    """
    instance_name = Path(csv_path).stem
    logger.info(f"\n=== {instance_name} ===")

    try:
        graph, depot_id, capacity = load_graph_from_csv(csv_path)
    except Exception as exc:
        logger.error(f"Failed to load {csv_path}: {exc}")
        return {}

    results = {}

    stop_desc = (
        f"solution_limit={solution_limit}" if solution_limit is not None else ""
    )
    if time_limit is not None:
        stop_desc += f"{', ' if stop_desc else ''}time_limit={time_limit}s"

    # ---- Uncoarsened ----
    logger.info(f"  Running Uncoarsened ORTools ({stop_desc})...")
    t0 = time.perf_counter()
    solver = ORToolsVRPTWSolver(
        graph, depot_id, capacity,
        time_limit_seconds=time_limit,
        solution_limit=solution_limit,
    )
    routes, metrics = solver.solve()
    metrics["computation_time"] = time.perf_counter() - t0
    results["Uncoarsened ORTools"] = metrics
    logger.info(
        f"  -> vehicles={metrics['num_vehicles']}, "
        f"distance={metrics['total_distance']:.2f}, "
        f"feasible={metrics['is_feasible']}, "
        f"TW violations={metrics['time_window_violations']}, "
        f"status={metrics.get('solver_status','?')}, "
        f"actual_time={metrics['computation_time']:.3f}s"
    )

    # ---- Coarsened warm-start pipeline ----
    # Timing starts here and covers the entire coarsened pipeline so that
    # the reported time is directly comparable to the uncoarsened time above.
    t0 = time.perf_counter()

    # Step 1: coarsen
    logger.info(f"  Coarsening graph...")
    coarsener = SpatioTemporalGraphCoarsener(
        graph=graph, depot_id=depot_id, **COARSENER_PARAMS
    )
    coarsened_graph, _ = coarsener.coarsen()
    n_orig = len(graph.nodes) - 1
    n_coarsened = len(coarsened_graph.nodes) - 1
    logger.info(f"  -> {n_orig} customers -> {n_coarsened} super-nodes")

    # Step 2: solve on coarsened graph.
    # Uses the same stopping criterion as uncoarsened — fair comparison.
    logger.info(f"  Step 2 — Solving on coarsened graph ({stop_desc})...")
    solver_c = ORToolsVRPTWSolver(
        coarsened_graph, depot_id, capacity,
        time_limit_seconds=time_limit,
        solution_limit=solution_limit,
    )
    coarsened_routes, _ = solver_c.solve()

    # Step 3: inflate back to original node IDs
    inflated_routes = coarsener.inflate_route(coarsened_routes)
    metrics_raw = calculate_route_metrics(graph, inflated_routes, depot_id, capacity)
    logger.info(
        f"  Step 3 — Raw inflated: vehicles={metrics_raw['num_vehicles']}, "
        f"distance={metrics_raw['total_distance']:.2f}, "
        f"TW violations={metrics_raw['time_window_violations']}"
    )

    # Step 4: warm-start repair on original graph.
    # Uses the same stopping criteria as the uncoarsened run so that
    # OR-Tools can fully optimise from the inflated warm-start.
    logger.info(f"  Step 4 — Warm-start repair on original graph ({stop_desc or 'no limit'})...")
    solver_w = ORToolsVRPTWSolver(
        graph, depot_id, capacity,
        time_limit_seconds=time_limit,
        solution_limit=solution_limit,
    )
    final_routes, metrics_inf = solver_w.solve(initial_routes=inflated_routes)
    metrics_inf["computation_time"] = time.perf_counter() - t0
    results["Inflated ORTools"] = metrics_inf
    logger.info(
        f"  -> vehicles={metrics_inf['num_vehicles']}, "
        f"distance={metrics_inf['total_distance']:.2f}, "
        f"feasible={metrics_inf['is_feasible']}, "
        f"TW violations={metrics_inf['time_window_violations']}, "
        f"status={metrics_inf.get('solver_status','?')}, "
        f"actual_time={metrics_inf['computation_time']:.3f}s"
    )

    return results


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary(all_results: dict) -> None:
    W = 92
    print("\n" + "=" * W)
    print(
        f"{'Instance':<12} {'Unc.Dist':>9} {'Inf.Dist':>9} {'Dist%':>7} "
        f"{'UncVeh':>6} {'InfVeh':>6} "
        f"{'UncTime':>8} {'InfTime':>8} {'Speedup':>8} "
        f"{'UncFeas':>8} {'InfFeas':>8}"
    )
    print("-" * W)

    dist_improvements, speedups = [], []
    for instance, res in sorted(all_results.items()):
        unc = res.get("Uncoarsened ORTools", {})
        inf = res.get("Inflated ORTools", {})
        ud  = unc.get("total_distance", 0.0)
        id_ = inf.get("total_distance", 0.0)
        ut  = unc.get("computation_time", 0.0)
        it  = inf.get("computation_time", 0.0)
        dist_imp = ((ud - id_) / ud * 100) if ud > 0 else 0.0
        speedup  = ut / it if it > 0 else float("inf")
        dist_improvements.append(dist_imp)
        if it > 0:
            speedups.append(speedup)
        name = Path(instance).stem
        print(
            f"{name:<12} {ud:>9.2f} {id_:>9.2f} {dist_imp:>+7.1f}% "
            f"{unc.get('num_vehicles',0):>6} {inf.get('num_vehicles',0):>6} "
            f"{ut:>8.3f}s {it:>8.3f}s {speedup:>7.1f}x "
            f"{'Yes' if unc.get('is_feasible') else 'No':>8} "
            f"{'Yes' if inf.get('is_feasible') else 'No':>8}"
        )

    print("-" * W)
    if dist_improvements:
        med_d = sorted(dist_improvements)[len(dist_improvements) // 2]
        print(f"Median distance change (inflated vs uncoarsened): {med_d:+.2f}%")
    if speedups:
        med_s = sorted(speedups)[len(speedups) // 2]
        print(f"Median speedup  (uncoarsened time / inflated time): {med_s:.1f}x")
    print("=" * W)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run OR-Tools VRPTW solver on Solomon benchmarks."
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
        default=str(Path(__file__).resolve().parent / "outputs" / "results_ortools.json"),
        help="Path for the output JSON file"
    )
    parser.add_argument(
        "--solution-limit", type=int, default=None,
        help="Stop after N improving solutions (default: no limit). "
             "OR-Tools exits immediately so actual time is measurable."
    )
    parser.add_argument(
        "--time-limit", type=int, default=None,
        help="Hard wall-clock cap in seconds per instance (default: no limit). "
             "Use when you need a quick run; omit to find optimal solutions."
    )
    args = parser.parse_args()
    solution_limit = args.solution_limit

    script_dir = Path(__file__).resolve().parent

    # Single-file mode
    if args.file:
        csv = Path(args.file)
        if not csv.is_file():
            logger.error(f"File not found: {csv}")
            return
        all_results = {str(csv): process_file(str(csv), solution_limit=solution_limit, time_limit=args.time_limit)}

    # Full-dataset mode
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
            instance_name = f.stem
            all_results[instance_name] = process_file(str(f), solution_limit=solution_limit, time_limit=args.time_limit)

    # Save JSON
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as fh:
        json.dump(all_results, fh, indent=4)
    logger.info(f"\nResults saved to {output_path}")

    # Print summary table
    print_summary(all_results)


if __name__ == "__main__":
    main()
