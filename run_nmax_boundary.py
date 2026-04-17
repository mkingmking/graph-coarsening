"""
N_max Boundary Experiment
=========================
Runs a single uncoarsened FullQuboSolver on one instance with increasing N.
Stops as soon as the solver wall-clock time exceeds TIME_LIMIT seconds.
Reports N_max — the largest N solved within the budget.

Usage:
    python -m graph_coarsening.run_nmax_boundary --file solomon_dataset/C1/C101.csv
    python -m graph_coarsening.run_nmax_boundary --file solomon_dataset/C1/C101.csv --start 17 --step 1 --max_n 100
"""

import argparse
import json
import time
from pathlib import Path

from .graph import compute_euclidean_tau
from .utils import load_graph_from_csv
from .quantum_solvers.vrp_problem import VRPProblem
from .quantum_solvers.vrp_solvers import FullQuboSolver

# QUBO params — identical to main_quantum.py
QUBO_PARAMS = {
    'only_one':          10_000_000,
    'capacity_penalty':   5_000_000,
    'time_window_penalty':3_000_000,
    'vehicle_start_cost':   100_000,
    'order':                    100,
    'backend':          'simulated',
    'reads':                   5000,
}

TIME_LIMIT = 1000  # seconds


def build_subgraph_vrp(full_graph, depot_id, capacity, num_customers):
    """Build a VRPProblem directly from the first num_customers nodes."""
    customer_ids = sorted(
        [nid for nid in full_graph.nodes if nid != depot_id], key=int
    )[:num_customers]

    int_to_id = [depot_id] + customer_ids
    id_to_int = {nid: i for i, nid in enumerate(int_to_id)}
    n = len(int_to_id)
    depot_int = id_to_int[depot_id]

    costs = [[0.0] * n for _ in range(n)]
    time_costs = [[0.0] * n for _ in range(n)]
    demands, time_windows, service_times = {}, {}, {}

    for u_id, u_node in full_graph.nodes.items():
        if u_id != depot_id and u_id not in id_to_int:
            continue
        u_int = id_to_int[u_id]
        demands[u_int]      = u_node.demand
        time_windows[u_int] = (u_node.e, u_node.l)
        service_times[u_int]= u_node.s
        for v_id, v_node in full_graph.nodes.items():
            if v_id != depot_id and v_id not in id_to_int:
                continue
            v_int = id_to_int[v_id]
            tau = 0.0 if u_id == v_id else compute_euclidean_tau(u_node, v_node)
            costs[u_int][v_int]      = tau
            time_costs[u_int][v_int] = tau

    customer_ints = [id_to_int[nid] for nid in customer_ids]
    vrp = VRPProblem(
        source_depot=depot_int,
        costs=costs,
        time_costs=time_costs,
        capacities=[capacity] * len(customer_ids),
        dests=customer_ints,
        weights=demands,
        time_windows=time_windows,
        service_times=service_times,
    )
    return vrp


def run_one(vrp: VRPProblem) -> float:
    """Solve and return wall-clock seconds."""
    solver = FullQuboSolver(vrp)
    t0 = time.perf_counter()
    solver.solve(
        QUBO_PARAMS['only_one'],
        QUBO_PARAMS['order'],
        QUBO_PARAMS['capacity_penalty'],
        QUBO_PARAMS['time_window_penalty'],
        QUBO_PARAMS['vehicle_start_cost'],
        QUBO_PARAMS['backend'],
        QUBO_PARAMS['reads'],
    )
    return time.perf_counter() - t0


def main():
    parser = argparse.ArgumentParser(description="Find N_max: largest uncoarsened N within time limit.")
    parser.add_argument("--file",   required=True,       help="Path to a Solomon CSV instance.")
    parser.add_argument("--start",  type=int, default=17, help="First N to test (default: 5).")
    parser.add_argument("--step",   type=int, default=1, help="Increment per step (default: 1).")
    parser.add_argument("--max_n",  type=int, default=100, help="Hard ceiling (default: 100).")
    parser.add_argument("--output", type=str, default=None, help="Optional JSON file to save all timings.")
    args = parser.parse_args()

    full_graph, depot_id, capacity = load_graph_from_csv(args.file)

    print("=" * 54)
    print(f" N_max Boundary Experiment — {Path(args.file).stem}")
    print(f" Solver : FullQuboSolver (uncoarsened)")
    print(f" N range: {args.start} → {args.max_n}  step={args.step}")
    print(f" Limit  : {TIME_LIMIT}s")
    print("=" * 54)

    timings = {}
    last_ok_n    = args.start - args.step
    last_ok_time = None

    for N in range(args.start, args.max_n + 1, args.step):
        print(f"  N={N:3d} ... ", end="", flush=True)

        vrp = build_subgraph_vrp(full_graph, depot_id, capacity, N)
        elapsed = run_one(vrp)
        timings[N] = round(elapsed, 2)

        status = "OK" if elapsed <= TIME_LIMIT else "OVER"
        print(f"{elapsed:7.1f}s  [{status}]")

        if elapsed > TIME_LIMIT:
            print(f"\n  Time limit exceeded at N={N} ({elapsed:.1f}s > {TIME_LIMIT}s).")
            break

        last_ok_n    = N
        last_ok_time = elapsed

    print()
    print("=" * 54)
    print(f" N_max = {last_ok_n}  (solved in {last_ok_time:.1f}s)")
    print("=" * 54)

    if args.output:
        payload = {
            "instance":    Path(args.file).stem,
            "solver":      "FullQuboSolver_uncoarsened",
            "time_limit":  TIME_LIMIT,
            "N_max":       last_ok_n,
            "timings":     timings,
        }
        with open(args.output, "w") as f:
            json.dump(payload, f, indent=4)
        print(f" Timings saved to {args.output}")


if __name__ == "__main__":
    main()
