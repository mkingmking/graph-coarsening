import os
import getpass
import logging
import json
import time
import argparse
from pathlib import Path

from ..graph import Graph, compute_euclidean_tau
from ..utils import load_graph_from_csv, calculate_route_metrics, resolve_coarsening_params
from ..coarsener import SpatioTemporalGraphCoarsener
from ..quantum_solvers.vrp_problem import VRPProblem
from ..quantum_solvers.vrp_solvers import FullQuboSolver, AveragePartitionSolver


def prompt_for_api_key() -> str:
    print("\n" + "="*60)
    print("  D-Wave Leap Quantum Solver")
    print("="*60)
    print("Enter your D-Wave Leap API token.")
    print("(It will not be echoed. Get it from cloud.dwavesys.com/leap)")
    print("-"*60)
    token = getpass.getpass("API token: ").strip()
    if not token:
        raise SystemExit("No API token provided. Exiting.")
    return token


def prompt_for_backend() -> str:
    print("\nSelect backend (press Enter for default):")
    print("  [1] hybrid  - LeapHybridSampler, works for any problem size  (default)")
    print("  [2] qpu     - Direct QPU, small instances only (<=5 customers)")
    choice = input("Choice [1/2]: ").strip()
    if choice == "2":
        print("Backend: qpu")
        return "qpu"
    print("Backend: hybrid")
    return "hybrid"


def convert_graph_to_vrp_problem_inputs(graph: Graph, depot_id: str, vehicle_capacity: float) -> tuple[VRPProblem, list]:
    customer_ids = sorted([nid for nid in graph.nodes if nid != depot_id])
    int_to_id_map = [depot_id] + customer_ids
    id_to_int_map = {nid: i for i, nid in enumerate(int_to_id_map)}

    num_nodes = len(int_to_id_map)
    int_depot_id = id_to_int_map[depot_id]

    costs = [[0.0] * num_nodes for _ in range(num_nodes)]
    time_costs = [[0.0] * num_nodes for _ in range(num_nodes)]
    demands = {}
    time_windows = {}
    service_times = {}

    for u_id, u_node in graph.nodes.items():
        u_int = id_to_int_map[u_id]
        demands[u_int] = u_node.demand
        time_windows[u_int] = (u_node.e, u_node.l)
        service_times[u_int] = u_node.s
        for v_id, v_node in graph.nodes.items():
            v_int = id_to_int_map[v_id]
            tau = 0.0 if u_id == v_id else compute_euclidean_tau(u_node, v_node)
            costs[u_int][v_int] = tau
            time_costs[u_int][v_int] = tau

    num_customers = len(customer_ids)
    num_vehicles = max(2, num_customers // 2)
    capacities = [vehicle_capacity] * num_vehicles
    customer_ints = [id_to_int_map[nid] for nid in customer_ids]

    vrp_problem = VRPProblem(
        source_depot=int_depot_id, costs=costs, time_costs=time_costs,
        capacities=capacities, dests=customer_ints, weights=demands,
        time_windows=time_windows, service_times=service_times
    )
    return vrp_problem, int_to_id_map


def map_solution_to_original_ids(solution_routes_int: list, int_to_id_map: list) -> list:
    mapped_routes = []
    for route_int in solution_routes_int:
        if route_int:
            mapped_routes.append([int_to_id_map[i] for i in route_int])
    return mapped_routes


def run_solver_pipeline(graph: Graph, depot_id: str, vehicle_capacity: float, solver_name: str, backend: str = 'hybrid', coarsener: SpatioTemporalGraphCoarsener = None):
    start_time = time.perf_counter()

    qubo_params = {
        'only_one': 10_000_000,
        'capacity_penalty': 5_000_000,
        'time_window_penalty': 3_000_000,
        'vehicle_start_cost': 100_000,
        'order': 100,
        'backend': backend,
        'reads': 1000          # used by qpu; ignored by hybrid
    }

    vrp, int_to_id_map = convert_graph_to_vrp_problem_inputs(graph, depot_id, vehicle_capacity)

    if solver_name == 'FullQubo':
        solver = FullQuboSolver(vrp)
    elif solver_name == 'AveragePartitionSolver':
        solver = AveragePartitionSolver(vrp)
    else:
        raise ValueError(f"Unknown solver: {solver_name}")

    sol = solver.solve(
        qubo_params['only_one'],
        qubo_params['order'],
        qubo_params['capacity_penalty'],
        qubo_params['time_window_penalty'],
        qubo_params['vehicle_start_cost'],
        qubo_params['backend'],
        qubo_params['reads']
    )

    solution_routes_str = map_solution_to_original_ids(sol.solution, int_to_id_map)

    formatted = []
    for r in solution_routes_str:
        if r:
            tmp = [depot_id] + r + [depot_id]
            if len(tmp) > 2:
                formatted.append(tmp)

    routes = formatted
    metrics_graph = graph
    if coarsener:
        routes = coarsener.inflate_route(formatted)
        metrics_graph = coarsener.graph

    metrics = calculate_route_metrics(metrics_graph, routes, depot_id, vehicle_capacity)

    end_time = time.perf_counter()
    duration = end_time - start_time
    return routes, metrics, duration


def configure_logging(level=logging.INFO):
    logging.basicConfig(level=level, format='%(levelname)s: %(message)s')
    return logging.getLogger(__name__)

logger = configure_logging()


def log_solver_results(prefix: str, routes: list, metrics: dict, duration: float):
    logger.info(f"\n--- {prefix} Results ---")
    logger.info(f"  Computation Time: {duration:.4f} seconds")
    logger.info(f"  Number of routes: {len(routes)}")

    all_customers = []
    for route in routes:
        customers_in_route = route[1:-1] if len(route) > 2 else []
        all_customers.extend(customers_in_route)

    unique_customers = set(all_customers)
    if len(all_customers) != len(unique_customers):
        duplicates = [c for c in all_customers if all_customers.count(c) > 1]
        logger.warning(f"  WARNING: Customers visited multiple times: {set(duplicates)}")
    else:
        logger.info(f"  All {len(unique_customers)} customers visited exactly once")

    for route_idx, route in enumerate(routes):
        logger.info(f"    Route {route_idx + 1}: {' -> '.join(str(n) for n in route)}")

    for k, v in metrics.items():
        if isinstance(v, float):
            logger.info(f"    {k.replace('_',' ').title()}: {v:.2f}")
        else:
            logger.info(f"    {k.replace('_',' ').title()}: {v}")


def final_summary(all_results: dict):
    logger.info("\n\n" + "="*25 + " FINAL SUMMARY " + "="*25)
    metrics_list = [
        "is_feasible", "total_distance", "num_vehicles",
        "total_route_duration", "computation_time"
    ]

    for fname, res in sorted(all_results.items()):
        logger.info(f"\n--- Results for {Path(fname).name} ---")
        for solver_name in ('FullQubo', 'AveragePartitionSolver'):
            uncoarsened_key = f"Uncoarsened {solver_name}"
            inflated_key = f"Inflated {solver_name}"

            if uncoarsened_key not in res or inflated_key not in res:
                continue

            logger.info(f"\n-- Comparison for {solver_name} --")
            uncoarsened_metrics = res[uncoarsened_key]
            inflated_metrics = res[inflated_key]

            logger.info(f"  {'Metric':<25} | {'Uncoarsened':<15} | {'Coarsened':<15}")
            logger.info(f"  {'-'*25} | {'-'*15} | {'-'*15}")

            for m in metrics_list:
                val_u = uncoarsened_metrics.get(m, 'N/A')
                val_i = inflated_metrics.get(m, 'N/A')
                if isinstance(val_u, float): val_u = f"{val_u:.2f}"
                if isinstance(val_i, float): val_i = f"{val_i:.2f}"
                logger.info(f"  {m.replace('_',' ').title():<25} | {str(val_u):<15} | {str(val_i):<15}")


def create_subgraph(original_graph: Graph, depot_id: str, num_customers: int) -> Graph:
    subgraph = Graph()
    subgraph.add_node(original_graph.nodes[depot_id])
    customer_ids = sorted([nid for nid in original_graph.nodes if nid != depot_id], key=int)
    customers_to_include = customer_ids[:num_customers]
    for cid in customers_to_include:
        subgraph.add_node(original_graph.nodes[cid])
    node_ids = list(subgraph.nodes.keys())
    for i in range(len(node_ids)):
        for j in range(i + 1, len(node_ids)):
            id1, id2 = node_ids[i], node_ids[j]
            original_edge = original_graph.get_edge_by_nodes(id1, id2)
            if original_edge:
                subgraph.add_edge(id1, id2, original_edge.tau)
    return subgraph


def process_file(csv_file_path: str, num_customers: int, backend: str = 'hybrid',
                 alpha: float = None, beta: float = None,
                 P: float = None, radiusCoeff: float = None) -> dict:
    logger.info(f"\n\n=== Processing file: {Path(csv_file_path).name} with {num_customers} customers ===")
    try:
        full_graph, depot_id, capacity = load_graph_from_csv(csv_file_path)
    except Exception as e:
        logger.error(f"Error loading {csv_file_path}: {e}")
        return {}

    subgraph = create_subgraph(full_graph, depot_id, num_customers)

    file_results = {}
    solvers_to_run = ('FullQubo', 'AveragePartitionSolver')

    for name in solvers_to_run:
        routes, metrics, duration = run_solver_pipeline(subgraph, depot_id, capacity, name, backend)
        metrics['computation_time'] = duration
        file_results[f"Uncoarsened {name}"] = metrics
        log_solver_results(f"Uncoarsened {name}", routes, metrics, duration)

    params = resolve_coarsening_params(csv_file_path, alpha=alpha, beta=beta, P=P, radiusCoeff=radiusCoeff)
    logger.info(f"Coarsening params: {params}")
    coarsener = SpatioTemporalGraphCoarsener(graph=subgraph, depot_id=depot_id, **params)
    coarsened_graph, _ = coarsener.coarsen()
    for name in solvers_to_run:
        routes, metrics, duration = run_solver_pipeline(coarsened_graph, depot_id, capacity, name, backend, coarsener)
        metrics['computation_time'] = duration
        file_results[f"Inflated {name}"] = metrics
        log_solver_results(f"Inflated {name}", routes, metrics, duration)

    return file_results


def main():
    parser = argparse.ArgumentParser(description="Run VRP solvers on D-Wave Leap hybrid hardware.")
    parser.add_argument("--file", type=str, default=None, help="Path to a single Solomon CSV file.")
    parser.add_argument("--data", type=str, default=None, help="Directory containing Solomon CSV files.")
    parser.add_argument("--customers", type=int, default=5, help="Number of customers (default: 5).")
    parser.add_argument("--output", type=str, default=None, help="Path to a JSON file to save results.")
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--beta", type=float, default=None)
    parser.add_argument("--P", type=float, default=None)
    parser.add_argument("--radius", type=float, default=None)
    args = parser.parse_args()

    # Ask for credentials and backend choice before any D-Wave sampler is instantiated.
    api_token = prompt_for_api_key()
    os.environ["DWAVE_API_TOKEN"] = api_token
    backend = prompt_for_backend()

    if args.file:
        files_to_process = [args.file]
        if not Path(args.file).is_file():
            logger.error(f"File not found: {args.file}")
            return
    else:
        script_dir = Path(__file__).resolve().parent.parent
        data_dir = Path(args.data) if args.data else script_dir / "solomon_dataset"
        if not data_dir.exists():
            logger.error(f"Data directory not found: {data_dir}")
            return
        files_to_process = [str(p) for p in sorted(data_dir.rglob("*.csv"))]
        if not files_to_process:
            logger.warning(f"No CSV files found in {data_dir}.")
            return

    logger.info("\n" + "="*60)
    logger.info(f"Backend: D-Wave Leap  [{backend}]  (real hardware)")
    logger.info("="*60)

    all_results = {}
    for csv_path in files_to_process:
        results = process_file(csv_path, args.customers, backend,
                               alpha=args.alpha, beta=args.beta, P=args.P, radiusCoeff=args.radius)
        all_results[csv_path] = results

    if args.output:
        output_path = Path(args.output)
    else:
        import datetime
        script_dir = Path(__file__).resolve().parent.parent
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = script_dir / "outputs" / f"results_dwave_{backend}_{args.customers}customers_{timestamp}.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=4)
    logger.info(f"\nResults saved to {output_path}")

    final_summary(all_results)
    logger.info("\nAll done.")


if __name__ == "__main__":
    main()
