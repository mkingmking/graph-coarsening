"""
Random hyperparameter search over coarsening parameters (P, radius, alpha, beta)
across multiple Solomon instances. Finds the best per-instance and globally best
hyperparameter set.

Uses simulated annealing so runs are cheap — then use the found hyperparameters
with --backend hybrid for the real run.

Usage (from repo root):
    python3 -m graph_coarsening.run_hyperparam_search [options]

Examples:
    # All 56 datasets, 30 trials each (warning: ~2h with default num-reads)
    python -m graph_coarsening.run_hyperparam_search

    # Just C-type instances, faster scan
    python -m graph_coarsening.run_hyperparam_search --families C1 C2 --n-trials 40

    # Single instance (equivalent to original behaviour)
    python -m graph_coarsening.run_hyperparam_search --families C1 --n-trials 30

    # Fast scan with fewer reads (noisier but quick for a first pass)
    python -m graph_coarsening.run_hyperparam_search --num-reads 1000 --n-trials 50

    # Resume an interrupted run (skips already-completed instances)
    python -m graph_coarsening.run_hyperparam_search --resume

Options:
    --solver        {fullqubo, averagepartition, greedy, savings}  (default: fullqubo)
    --customers     INT     Customers per instance (default: 10)
    --families      STR ... Limit to these dataset families, e.g. C1 C2 R1
                            (default: all families — C1 C2 R1 R2 RC1 RC2)
    --n-trials      INT     Random combos to try per instance (default: 30)
    --seed          INT     Random seed — same seed = same trial list for all
                            instances, enabling fair cross-instance comparison
                            (default: 42)
    --num-reads     INT     SA reads per trial (default: 5000)
    --output        PATH    JSON checkpoint/results file (default:
                            hyperparam_search_results.json)
    --resume                If set, skip instances already present in --output
"""

import argparse
import json
import random
import time
from pathlib import Path

from ..graph import Graph, compute_euclidean_tau
from ..utils import load_graph_from_csv, calculate_route_metrics
from ..coarsener import SpatioTemporalGraphCoarsener
from ..quantum_solvers.vrp_problem import VRPProblem
from ..quantum_solvers.vrp_solvers import FullQuboSolver, AveragePartitionSolver
from ..greedy_solver import GreedySolver
from ..savings_solver import SavingsSolver

CLASSICAL_SOLVERS = {"greedy": GreedySolver, "savings": SavingsSolver}
QUANTUM_SOLVERS   = {"fullqubo": FullQuboSolver, "averagepartition": AveragePartitionSolver}

# ── Search space ───────────────────────────────────────────────────────────────

P_VALUES      = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
RADIUS_VALUES = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]
ALPHA_VALUES  = [0.3, 0.5, 0.7, 0.9, 1.0]
BETA_VALUES   = [0.2, 0.4, 0.6, 0.8, 1.0]

QUBO_PARAMS = {
    "only_one":    10_000_000,
    "order":       100,
    "cap_penalty": 5_000_000,
    "tw_penalty":  3_000_000,
    "start_cost":  100_000,
    "backend":     "simulated",
}

ALL_FAMILIES = ["C1", "C2", "R1", "R2", "RC1", "RC2"]

# ── Dataset discovery ─────────────────────────────────────────────────────────

def discover_datasets(pkg_dir: Path, families: list[str]) -> list[Path]:
    paths = []
    for family in families:
        folder = pkg_dir / "solomon_dataset" / family
        if folder.exists():
            paths.extend(sorted(folder.glob("*.csv")))
    return paths


# ── Shared helpers ─────────────────────────────────────────────────────────────

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
    capacities = [vehicle_capacity] * max(2, num_customers // 2)
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


def _coverage(routes, graph, depot_id):
    all_customers = set(nid for nid in graph.nodes if nid != depot_id)
    visited = set(nid for route in routes for nid in route if nid != depot_id)
    return len(all_customers - visited)


# ── Single trial ───────────────────────────────────────────────────────────────

def _run_trial(graph, depot_id, capacity, P, radius, alpha, beta, solver_cls, num_reads):
    coarsener = SpatioTemporalGraphCoarsener(
        graph=graph, alpha=alpha, beta=beta, P=P, radiusCoeff=radius, depot_id=depot_id
    )
    coarsened_graph, _ = coarsener.coarsen()
    n_coarsened = len(coarsened_graph.nodes) - 1

    vrp, int_to_id = _graph_to_vrp_problem(coarsened_graph, depot_id, capacity)
    solver = solver_cls(vrp)
    sol = solver.solve(
        QUBO_PARAMS["only_one"], QUBO_PARAMS["order"],
        QUBO_PARAMS["cap_penalty"], QUBO_PARAMS["tw_penalty"],
        QUBO_PARAMS["start_cost"], QUBO_PARAMS["backend"], num_reads,
    )
    routes = _map_to_str_routes(sol.solution, int_to_id, depot_id)
    inflated = coarsener.inflate_route(routes)
    metrics = calculate_route_metrics(graph, inflated, depot_id, capacity)
    if _coverage(inflated, graph, depot_id) > 0:
        metrics["is_feasible"] = False
    return {
        "feasible":        bool(metrics.get("is_feasible", False)),
        "total_distance":  metrics.get("total_distance", None),
        "num_vehicles":    metrics.get("num_vehicles", None),
        "tw_violations":   metrics.get("time_window_violations", 0),
        "n_coarsened":     n_coarsened,
    }


# ── Classical single trial ─────────────────────────────────────────────────────

def _run_classical_trial(graph, depot_id, capacity, P, radius, alpha, beta, solver_cls):
    """Run one trial for a classical (Greedy/Savings) solver.
    Coarsens the graph, solves on the coarsened graph, inflates, and evaluates
    feasibility on the original graph — same criterion as quantum trials.
    """
    coarsener = SpatioTemporalGraphCoarsener(
        graph=graph, alpha=alpha, beta=beta, P=P, radiusCoeff=radius, depot_id=depot_id
    )
    coarsened_graph, _ = coarsener.coarsen()
    n_coarsened = len(coarsened_graph.nodes) - 1

    solver = solver_cls(coarsened_graph, depot_id, capacity)
    raw_routes, _ = solver.solve()

    # Format routes and inflate back to original customer set
    formatted = []
    for route in raw_routes:
        if route:
            tmp = [depot_id] + route + [depot_id]
            if len(tmp) > 2:
                formatted.append(tmp)
    inflated = coarsener.inflate_route(formatted)

    metrics = calculate_route_metrics(graph, inflated, depot_id, capacity)
    if _coverage(inflated, graph, depot_id) > 0:
        metrics["is_feasible"] = False
    return {
        "feasible":        bool(metrics.get("is_feasible", False)),
        "total_distance":  metrics.get("total_distance", None),
        "num_vehicles":    metrics.get("num_vehicles", None),
        "tw_violations":   metrics.get("time_window_violations", 0),
        "n_coarsened":     n_coarsened,
    }


# ── Per-instance search ────────────────────────────────────────────────────────

def run_instance(graph, depot_id, capacity, solver_cls, trials, num_reads, is_classical=False):
    """Run all trials on one instance. Returns list of result dicts."""
    results = []
    n = len(trials)
    for idx, (P, radius, alpha, beta) in enumerate(trials, 1):
        t0 = time.perf_counter()
        try:
            if is_classical:
                m = _run_classical_trial(graph, depot_id, capacity, P, radius, alpha, beta,
                                         solver_cls)
            else:
                m = _run_trial(graph, depot_id, capacity, P, radius, alpha, beta,
                               solver_cls, num_reads)
            elapsed = round(time.perf_counter() - t0, 2)
            status = "✓" if m["feasible"] else "✗"
            dist_str = f"{m['total_distance']:.1f}" if m["total_distance"] else "—"
            print(f"  [{idx:>3}/{n}] {status} P={P} r={radius} a={alpha} b={beta}"
                  f"  dist={dist_str}  coars→{m['n_coarsened']}  ({elapsed}s)")
        except Exception as e:
            print(f"  [{idx:>3}/{n}] ERROR: {e}")
            m = {"feasible": False, "total_distance": None, "num_vehicles": None,
                 "tw_violations": None, "n_coarsened": None}
            elapsed = round(time.perf_counter() - t0, 2)

        results.append({
            "P": P, "radius": radius, "alpha": alpha, "beta": beta,
            "feasible":       m["feasible"],
            "total_distance": m["total_distance"],
            "num_vehicles":   m["num_vehicles"],
            "tw_violations":  m["tw_violations"],
            "n_coarsened":    m["n_coarsened"],
            "elapsed":        elapsed,
        })
    return results


# ── Aggregation ────────────────────────────────────────────────────────────────

def _combo_key(r):
    return (r["P"], r["radius"], r["alpha"], r["beta"])


def aggregate(all_instance_results: dict) -> list:
    """
    all_instance_results: {instance_name: [trial_result, ...]}

    Returns list of combo dicts sorted by:
      1. total_feasible (across all instances) descending
      2. instances_with_any_feasible descending
    """
    combo_stats: dict[tuple, dict] = {}

    for instance_name, trials in all_instance_results.items():
        for r in trials:
            key = _combo_key(r)
            if key not in combo_stats:
                combo_stats[key] = {
                    "P": r["P"], "radius": r["radius"],
                    "alpha": r["alpha"], "beta": r["beta"],
                    "total_feasible": 0,
                    "instances_with_feasible": 0,
                    "per_instance": {},
                }
            s = combo_stats[key]
            if r["feasible"]:
                s["total_feasible"] += 1
                s["per_instance"][instance_name] = r["total_distance"]
            if instance_name not in s["per_instance"]:
                s["per_instance"][instance_name] = None  # ran but not feasible

    # Count instances where this combo was feasible
    for s in combo_stats.values():
        s["instances_with_feasible"] = sum(
            1 for v in s["per_instance"].values() if v is not None
        )

    ranked = sorted(
        combo_stats.values(),
        key=lambda s: (-s["total_feasible"], -s["instances_with_feasible"]),
    )
    return ranked


def best_per_instance(all_instance_results: dict) -> dict:
    """Returns {instance_name: best_combo_dict} ranked by feasible first."""
    out = {}
    for instance_name, trials in all_instance_results.items():
        feasible = [r for r in trials if r["feasible"]]
        if feasible:
            best = min(feasible, key=lambda r: r["total_distance"] or float("inf"))
        else:
            # No feasible — pick least-bad (fewest TW violations)
            best = min(trials, key=lambda r: (r["tw_violations"] or 999,
                                              r["total_distance"] or float("inf")))
        out[instance_name] = best
    return out


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Random hyperparameter search across all Solomon instances.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--solver", choices=["fullqubo", "averagepartition", "greedy", "savings"],
                        default="fullqubo")
    parser.add_argument("--customers", type=int, default=10)
    parser.add_argument("--families", nargs="+", default=ALL_FAMILIES,
                        choices=ALL_FAMILIES, metavar="FAMILY",
                        help="Dataset families to include (e.g. C1 C2 R1)")
    parser.add_argument("--n-trials", type=int, default=30,
                        help="Random combos to try per instance")
    parser.add_argument("--seed", type=int, default=42,
                        help="Same seed = same trial list for every instance")
    parser.add_argument("--num-reads", type=int, default=5_000)
    parser.add_argument("--output", default="hyperparam_search_results.json")
    parser.add_argument("--resume", action="store_true",
                        help="Skip instances already present in --output")
    args = parser.parse_args()

    pkg_dir = Path(__file__).resolve().parent.parent
    is_classical = args.solver in CLASSICAL_SOLVERS
    solver_cls = (CLASSICAL_SOLVERS if is_classical else QUANTUM_SOLVERS)[args.solver]

    # ── Discover datasets ─────────────────────────────────────────
    csv_paths = discover_datasets(pkg_dir, args.families)
    if not csv_paths:
        print("No datasets found. Check --families and solomon_dataset/ folder.")
        return

    # ── Generate trial list (same for every instance) ─────────────
    rng = random.Random(args.seed)
    all_combos = [
        (P, r, a, b)
        for P in P_VALUES
        for r in RADIUS_VALUES
        for a in ALPHA_VALUES
        for b in BETA_VALUES
    ]
    rng.shuffle(all_combos)
    trials = all_combos[:args.n_trials]

    total_runs = len(csv_paths) * args.n_trials
    print(f"\nSolver      : {solver_cls.__name__}  ({'classical' if is_classical else 'quantum'})")
    print(f"Customers   : {args.customers}")
    print(f"Families    : {args.families}")
    print(f"Instances   : {len(csv_paths)}")
    print(f"Trials/inst : {args.n_trials}")
    if not is_classical:
        print(f"SA reads    : {args.num_reads}")
    print(f"Total runs  : {total_runs}")
    print(f"Output      : {pkg_dir / args.output}")

    # ── Load checkpoint if resuming ───────────────────────────────
    output_path = pkg_dir / args.output
    all_results: dict = {}
    if args.resume and output_path.exists():
        with open(output_path) as f:
            saved = json.load(f)
        all_results = saved.get("per_instance_trials", {})
        print(f"\nResuming: {len(all_results)} instances already done, "
              f"{len(csv_paths) - len(all_results)} remaining.")

    # ── Run per instance ──────────────────────────────────────────
    for i, csv_path in enumerate(csv_paths, 1):
        instance_name = csv_path.stem
        if instance_name in all_results:
            print(f"\n[{i}/{len(csv_paths)}] {instance_name}  (skipped — already done)")
            continue

        print(f"\n{'='*64}")
        print(f"[{i}/{len(csv_paths)}]  {instance_name}")
        print(f"{'='*64}")

        graph, depot_id, capacity = load_graph_from_csv(str(csv_path))
        graph = _create_subgraph(graph, depot_id, args.customers)

        instance_results = run_instance(
            graph, depot_id, capacity, solver_cls, trials, args.num_reads,
            is_classical=is_classical
        )
        all_results[instance_name] = instance_results

        feasible_count = sum(1 for r in instance_results if r["feasible"])
        print(f"  → {feasible_count}/{args.n_trials} trials feasible")

        # Checkpoint after every instance
        _save(output_path, args, solver_cls, trials, all_results)

    # ── Aggregate ─────────────────────────────────────────────────
    ranked_global = aggregate(all_results)
    per_inst_best  = best_per_instance(all_results)

    # ── Print global top-10 ───────────────────────────────────────
    n_inst = len(all_results)
    print(f"\n\n{'='*80}")
    print(f"  GLOBAL BEST HYPERPARAMETERS  ({n_inst} instances × {args.n_trials} trials)")
    print(f"{'='*80}")
    print(f"  {'Rank':<5} {'P':<6} {'Radius':<8} {'Alpha':<7} {'Beta':<7} "
          f"{'Total✓':>8} {'Instances✓':>12}")
    print(f"  {'-'*5} {'-'*6} {'-'*8} {'-'*7} {'-'*7} {'-'*8} {'-'*12}")
    for rank, s in enumerate(ranked_global[:10], 1):
        print(f"  {rank:<5} {s['P']:<6} {s['radius']:<8} {s['alpha']:<7} {s['beta']:<7} "
              f"{s['total_feasible']:>8} {s['instances_with_feasible']:>12}/{n_inst}")

    best = ranked_global[0]
    print(f"\n  ★ OVERALL BEST:")
    print(f"    P={best['P']}  radius={best['radius']}  "
          f"alpha={best['alpha']}  beta={best['beta']}")
    print(f"    Feasible in {best['instances_with_feasible']}/{n_inst} instances "
          f"({best['total_feasible']} total feasible runs)")

    # ── Print per-instance summary ────────────────────────────────
    print(f"\n\n{'='*80}")
    print(f"  PER-INSTANCE BEST")
    print(f"{'='*80}")
    print(f"  {'Instance':<10} {'Feasible?':>10} {'P':<6} {'Radius':<8} "
          f"{'Alpha':<7} {'Beta':<7} {'Distance':>10}")
    print(f"  {'-'*10} {'-'*10} {'-'*6} {'-'*8} {'-'*7} {'-'*7} {'-'*10}")
    for inst, b in sorted(per_inst_best.items()):
        dist = f"{b['total_distance']:.2f}" if b["total_distance"] else "—"
        print(f"  {inst:<10} {'Yes' if b['feasible'] else 'No':>10} "
              f"{b['P']:<6} {b['radius']:<8} {b['alpha']:<7} {b['beta']:<7} {dist:>10}")

    # ── Print hybrid run command for best combo ───────────────────
    print(f"\n  To run with hybrid solver using the best combo:")
    print(f"    python -m graph_coarsening.run_p_experiment \\")
    print(f"      --solver fullqubo --customers {args.customers} \\")
    print(f"      --p-values {best['P']} \\")
    print(f"      --alpha {best['alpha']} --beta {best['beta']} "
          f"--radius {best['radius']} \\")
    print(f"      --backend hybrid")

    # ── Final save with aggregation ───────────────────────────────
    _save(output_path, args, solver_cls, trials, all_results,
          global_ranking=ranked_global[:20],
          per_instance_best=per_inst_best)
    print(f"\nFull results saved to: {output_path}\n")


def _save(path, args, solver_cls, trials, all_results,
          global_ranking=None, per_instance_best=None):
    payload = {
        "config": {
            "solver":     solver_cls.__name__,
            "customers":  args.customers,
            "n_trials":   args.n_trials,
            "num_reads":  args.num_reads,
            "seed":       args.seed,
            "families":   args.families,
            "trial_list": [list(t) for t in trials],
        },
        "per_instance_trials": all_results,
    }
    if global_ranking is not None:
        # Remove bulky per_instance dict for top-level ranking readability
        payload["global_ranking_top20"] = [
            {k: v for k, v in s.items() if k != "per_instance"}
            for s in global_ranking
        ]
    if per_instance_best is not None:
        payload["per_instance_best"] = per_instance_best

    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


if __name__ == "__main__":
    main()
