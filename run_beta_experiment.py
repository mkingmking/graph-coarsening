"""
Experiment: Compare solver across beta values on a Solomon instance.

Usage (from repo root):
    python -m graph_coarsening.run_beta_experiment [options]

Examples:
    # Defaults: Greedy, C101, beta configs = 0.2 / 0.4 / 0.6 / 0.8 / 1.0
    python -m graph_coarsening.run_beta_experiment

    # Savings solver
    python -m graph_coarsening.run_beta_experiment --solver savings

    # Quantum — FullQubo on first 5 customers
    python -m graph_coarsening.run_beta_experiment --solver fullqubo --customers 5

    # Custom beta sweep + custom coarsening hyperparameters
    python -m graph_coarsening.run_beta_experiment --beta-values 0.3 0.6 0.9 \\
        --p 0.5 --alpha 0.8 --radius 2.0

    # Different dataset file
    python -m graph_coarsening.run_beta_experiment --csv solomon_dataset/R1/R101.csv

Options (general):
    --solver       {greedy,savings,fullqubo,averagepartition}
    --csv          PATH      Path to CSV relative to package dir (default: C1/C101.csv)
    --customers    INT       Restrict to first N customers (required for quantum solvers on large instances)
    --beta-values  FLOAT ... Temporal weight values to sweep (default: 0.2 0.4 0.6 0.8 1.0)
    --p            FLOAT     Coarsening ratio (default: 0.5)
    --alpha        FLOAT     Spatial weight for coarsener (default: 0.8)
    --radius       FLOAT     Radius coefficient for coarsener (default: 2.0)

Options (quantum only):
    --only-one     INT       Unique-visit constraint penalty (default: 10_000_000)
    --order        INT       Travel distance weight (default: 100)
    --cap-penalty  INT       Capacity constraint penalty (default: 5_000_000)
    --tw-penalty   INT       Time-window constraint penalty (default: 3_000_000)
    --start-cost   INT       Vehicle start cost (default: 100_000)
    --backend      STR       Sampler backend: simulated | dwave (default: simulated)
    --num-reads    INT       Number of sampler reads (default: 5_000)
"""

import argparse
import time
from pathlib import Path

from .graph import Graph, compute_euclidean_tau
from .utils import load_graph_from_csv, calculate_route_metrics
from .greedy_solver import GreedySolver
from .savings_solver import SavingsSolver
from .coarsener import SpatioTemporalGraphCoarsener
from .visualisation import visualize_routes
from .quantum_solvers.vrp_problem import VRPProblem
from .quantum_solvers.vrp_solvers import FullQuboSolver, AveragePartitionSolver

# ── Metrics to display ─────────────────────────────────────────────────────────

METRICS_TO_PRINT = [
    "total_distance",
    "num_vehicles",
    "unserved_customers",
    "total_route_duration",
    "time_window_violations",
    "capacity_violations",
    "is_feasible",
    "computation_time",
]

# ── Subgraph helper ────────────────────────────────────────────────────────────

def _create_subgraph(original_graph: Graph, depot_id: str, num_customers: int) -> Graph:
    subgraph = Graph()
    subgraph.add_node(original_graph.nodes[depot_id])
    customer_ids = sorted([nid for nid in original_graph.nodes if nid != depot_id], key=int)
    for cid in customer_ids[:num_customers]:
        subgraph.add_node(original_graph.nodes[cid])
    node_ids = list(subgraph.nodes.keys())
    for i in range(len(node_ids)):
        for j in range(i + 1, len(node_ids)):
            id1, id2 = node_ids[i], node_ids[j]
            edge = original_graph.get_edge_by_nodes(id1, id2)
            if edge:
                subgraph.add_edge(id1, id2, edge.tau)
    return subgraph


# ── VRPProblem conversion ──────────────────────────────────────────────────────

def _graph_to_vrp_problem(graph: Graph, depot_id: str, vehicle_capacity: float):
    customer_ids = sorted([nid for nid in graph.nodes if nid != depot_id])
    int_to_id = [depot_id] + customer_ids
    id_to_int = {nid: i for i, nid in enumerate(int_to_id)}

    num_nodes = len(int_to_id)
    int_depot = id_to_int[depot_id]

    costs = [[0.0] * num_nodes for _ in range(num_nodes)]
    time_costs = [[0.0] * num_nodes for _ in range(num_nodes)]
    demands, time_windows, service_times = {}, {}, {}

    for u_id, u_node in graph.nodes.items():
        u = id_to_int[u_id]
        demands[u] = u_node.demand
        time_windows[u] = (u_node.e, u_node.l)
        service_times[u] = u_node.s
        for v_id, v_node in graph.nodes.items():
            v = id_to_int[v_id]
            tau = 0.0 if u_id == v_id else compute_euclidean_tau(u_node, v_node)
            costs[u][v] = tau
            time_costs[u][v] = tau

    num_customers = len(customer_ids)
    num_vehicles = max(2, num_customers // 2)
    capacities = [vehicle_capacity] * num_vehicles
    customer_ints = [id_to_int[nid] for nid in customer_ids]

    vrp = VRPProblem(
        source_depot=int_depot, costs=costs, time_costs=time_costs,
        capacities=capacities, dests=customer_ints, weights=demands,
        time_windows=time_windows, service_times=service_times,
    )
    return vrp, int_to_id


def _map_to_str_routes(solution_routes_int, int_to_id, depot_id):
    formatted = []
    for route_int in solution_routes_int:
        if route_int:
            tmp = [depot_id] + [int_to_id[i] for i in route_int] + [depot_id]
            if len(tmp) > 2:
                formatted.append(tmp)
    return formatted


# ── Solver runner factories ────────────────────────────────────────────────────

def _classical_runner(cls):
    def run(graph, depot_id, capacity):
        solver = cls(graph, depot_id, capacity)
        routes, metrics = solver.solve()
        return routes, metrics
    run.__name__ = cls.__name__
    return run


def _quantum_runner(cls, qubo_params):
    def run(graph, depot_id, capacity, qubo_params=qubo_params):
        vrp, int_to_id = _graph_to_vrp_problem(graph, depot_id, capacity)
        solver = cls(vrp)
        sol = solver.solve(
            qubo_params["only_one"],
            qubo_params["order"],
            qubo_params["cap_penalty"],
            qubo_params["tw_penalty"],
            qubo_params["start_cost"],
            qubo_params["backend"],
            qubo_params["num_reads"],
        )
        routes = _map_to_str_routes(sol.solution, int_to_id, depot_id)
        metrics = calculate_route_metrics(graph, routes, depot_id, capacity)
        return routes, metrics
    run.__name__ = cls.__name__
    return run


# ── Coverage helper ────────────────────────────────────────────────────────────

def _coverage(routes, graph, depot_id):
    all_customers = set(nid for nid in graph.nodes if nid != depot_id)
    visited = set(nid for route in routes for nid in route if nid != depot_id)
    return len(all_customers - visited)


# ── Per-beta runner ────────────────────────────────────────────────────────────

def run_for_beta(graph, depot_id, capacity, beta, P, solver_runner, alpha, radius, instance_name):
    print(f"\n{'='*60}")
    print(f"  Beta={beta}  (P={P}  alpha={alpha}  radius={radius})")
    print(f"{'='*60}")

    solver_name = solver_runner.__name__.replace("Solver", "")

    coarsener = SpatioTemporalGraphCoarsener(
        graph=graph, alpha=alpha, beta=beta, P=P, radiusCoeff=radius, depot_id=depot_id
    )
    coarsened_graph, _ = coarsener.coarsen()
    n_orig = len(graph.nodes) - 1
    n_coarsened = len(coarsened_graph.nodes) - 1
    print(f"  Customers: {n_orig} → {n_coarsened}  (coarsened)")

    t0 = time.perf_counter()
    coarsened_routes, _ = solver_runner(coarsened_graph, depot_id, capacity)
    inflated_routes = coarsener.inflate_route(coarsened_routes)
    metrics = calculate_route_metrics(graph, inflated_routes, depot_id, capacity)
    metrics["computation_time"] = time.perf_counter() - t0
    metrics["unserved_customers"] = _coverage(inflated_routes, graph, depot_id)
    if metrics["unserved_customers"] > 0:
        metrics["is_feasible"] = False

    fname = f"{instance_name}_{solver_name}_beta{beta}_P{P}.png"
    visualize_routes(
        graph, inflated_routes, depot_id,
        title=f"{instance_name}  {solver_name}  |  beta={beta}  P={P}",
        filename=fname,
    )
    print(f"  Saved: visualisation_routes/{fname}")
    return metrics, n_coarsened


# ── Main ───────────────────────────────────────────────────────────────────────

QUANTUM_SOLVERS = {"fullqubo", "averagepartition"}

def main():
    parser = argparse.ArgumentParser(
        description="Compare solver across beta values on a Solomon VRPTW instance.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── General ───────────────────────────────────────────────────
    parser.add_argument(
        "--solver",
        choices=["greedy", "savings", "fullqubo", "averagepartition"],
        default="greedy",
        help="Solver to use",
    )
    parser.add_argument(
        "--csv", default=None,
        help="CSV path relative to package dir (default: solomon_dataset/C1/C101.csv)",
    )
    parser.add_argument(
        "--customers", type=int, default=None,
        help="Restrict graph to first N customers (recommended for quantum solvers)",
    )
    parser.add_argument(
        "--beta-values", nargs="+", type=float, default=[0.2, 0.4, 0.6, 0.8, 1.0],
        metavar="BETA",
        help="Temporal weight values to sweep",
    )

    # ── Fixed coarsening hyperparameters ──────────────────────────
    parser.add_argument("--p",      type=float, default=0.5, help="Coarsening ratio (fixed)")
    parser.add_argument("--alpha",  type=float, default=0.8, help="Spatial weight")
    parser.add_argument("--radius", type=float, default=2.0, help="Radius coefficient")

    # ── QUBO hyperparameters (quantum only) ───────────────────────
    parser.add_argument("--only-one",    type=int,   default=10_000_000, help="Unique-visit penalty")
    parser.add_argument("--order",       type=int,   default=100,        help="Travel distance weight")
    parser.add_argument("--cap-penalty", type=int,   default=5_000_000,  help="Capacity penalty")
    parser.add_argument("--tw-penalty",  type=int,   default=3_000_000,  help="Time-window penalty")
    parser.add_argument("--start-cost",  type=int,   default=100_000,    help="Vehicle start cost")
    parser.add_argument("--backend",     type=str,   default="simulated", help="simulated | dwave")
    parser.add_argument("--num-reads",   type=int,   default=5_000,      help="Sampler reads")

    args = parser.parse_args()

    # ── Resolve CSV path ──────────────────────────────────────────
    pkg_dir = Path(__file__).resolve().parent
    csv_rel = args.csv if args.csv else "solomon_dataset/C1/C101.csv"
    csv_path = pkg_dir / csv_rel
    instance_name = csv_path.stem

    # ── Build solver runner ───────────────────────────────────────
    if args.solver in QUANTUM_SOLVERS:
        if args.customers is None:
            parser.error(
                f"--customers is required for quantum solver '{args.solver}' "
                "(quantum solvers are expensive; specify e.g. --customers 5)"
            )
        qubo_params = {
            "only_one":    args.only_one,
            "order":       args.order,
            "cap_penalty": args.cap_penalty,
            "tw_penalty":  args.tw_penalty,
            "start_cost":  args.start_cost,
            "backend":     args.backend,
            "num_reads":   args.num_reads,
        }
        cls = FullQuboSolver if args.solver == "fullqubo" else AveragePartitionSolver
        solver_runner = _quantum_runner(cls, qubo_params)
    else:
        cls = GreedySolver if args.solver == "greedy" else SavingsSolver
        solver_runner = _classical_runner(cls)

    # ── Print config ──────────────────────────────────────────────
    print(f"\nInstance   : {instance_name}  ({csv_path})")
    print(f"Solver     : {solver_runner.__name__}")
    print(f"Beta values: {args.beta_values}")
    print(f"Coarsener  : P={args.p}  alpha={args.alpha}  radius={args.radius}")
    if args.solver in QUANTUM_SOLVERS:
        print(f"Customers  : {args.customers}  (subgraph)")
        print(f"QUBO       : only_one={args.only_one}  order={args.order}  "
              f"cap={args.cap_penalty}  tw={args.tw_penalty}  "
              f"start={args.start_cost}  backend={args.backend}  reads={args.num_reads}")

    # ── Load graph ────────────────────────────────────────────────
    graph, depot_id, capacity = load_graph_from_csv(str(csv_path))
    if args.customers is not None:
        graph = _create_subgraph(graph, depot_id, args.customers)
    print(f"\nLoaded: {len(graph.nodes) - 1} customers  |  depot={depot_id}  |  capacity={capacity}")

    # ── Run experiment ────────────────────────────────────────────
    all_results = {}
    for beta in args.beta_values:
        metrics, n_coarsened = run_for_beta(
            graph, depot_id, capacity, beta, args.p,
            solver_runner, args.alpha, args.radius, instance_name,
        )
        all_results[beta] = (metrics, n_coarsened)

        print(f"\n  Metrics:")
        for k in METRICS_TO_PRINT:
            v = metrics.get(k, "N/A")
            fmt = f"{v:.3f}" if isinstance(v, float) else str(v)
            print(f"    {k:<30}: {fmt}")

    # ── Comparison table ──────────────────────────────────────────
    solver_name = solver_runner.__name__.replace("Solver", "")
    print(f"\n\n{'='*72}")
    print(f"  COMPARISON SUMMARY — {instance_name}  {solver_name} Solver  (P={args.p})")
    print(f"{'='*72}")
    header = (
        f"  {'Beta':<8} {'N_coarsened':>12} {'Distance':>12} {'Vehicles':>10}"
        f" {'Unserved':>10} {'TW Viol':>8} {'Feasible':>9}"
    )
    print(header)
    print(f"  {'-'*8} {'-'*12} {'-'*12} {'-'*10} {'-'*10} {'-'*8} {'-'*9}")
    for beta, (m, n_coarsened) in all_results.items():
        print(
            f"  {beta:<8} {n_coarsened:>12} "
            f"{m.get('total_distance', 0):>12.2f} "
            f"{m.get('num_vehicles', 0):>10} "
            f"{m.get('unserved_customers', 0):>10} "
            f"{m.get('time_window_violations', 0):>8} "
            f"{str(m.get('is_feasible', False)):>9}"
        )
    print(f"{'='*72}\n")


if __name__ == "__main__":
    main()
