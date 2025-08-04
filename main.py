import os
import logging


from graph import Graph, compute_euclidean_tau
from utils import load_graph_from_csv, calculate_route_metrics
from greedy_solver import GreedySolver
from savings_solver import SavingsSolver
from coarsener import SpatioTemporalGraphCoarsener
from quantum_solvers.vrp_problem_qubo import VRPProblem
from quantum_solvers.vrp_solution import VRPSolution
from quantum_solvers.vrp_solvers import FullQuboSolver, AveragePartitionSolver
# Mock D-Wave solver for local testing
from quantum_solvers.mock_dwave_solvers import MockDWaveSolvers as DWaveSolvers_modified


def convert_graph_to_vrp_problem_inputs(graph: Graph, depot_id: str, vehicle_capacity: float) -> VRPProblem:
    costs_matrix = {u: {} for u in graph.nodes}
    time_costs_matrix = {u: {} for u in graph.nodes}
    for u_id, u_node in graph.nodes.items():
        for v_id, v_node in graph.nodes.items():
            tau = 0.0 if u_id == v_id else compute_euclidean_tau(u_node, v_node)
            costs_matrix[u_id][v_id] = tau
            time_costs_matrix[u_id][v_id] = tau
    num_vehicles = 3
    capacities = [vehicle_capacity] * num_vehicles
    customer_ids = [nid for nid in graph.nodes if nid != depot_id]
    customer_demands = {nid: graph.nodes[nid].demand for nid in customer_ids}
    return VRPProblem(graph.nodes, depot_id, costs_matrix, time_costs_matrix, capacities, customer_ids, customer_demands)


def run_solver_pipeline(graph: Graph, depot_id: str, vehicle_capacity: float, solver_name: str, coarsener: SpatioTemporalGraphCoarsener = None):
    qubo_params = dict(only_one=1000, order=1, tw_penalty=1000, reads=10, backend='simulated')
    if solver_name in ('Greedy', 'Savings'):
        solver = GreedySolver(graph, depot_id, vehicle_capacity) if solver_name == 'Greedy' else SavingsSolver(graph, depot_id, vehicle_capacity)
        routes, metrics = solver.solve()
        if coarsener:
            formatted = []
            for r in routes:
                if not r: continue
                tmp = [depot_id] + r + [depot_id]
                if len(tmp) > 2: formatted.append(tmp)
            routes = coarsener.inflate_route(formatted)
            metrics = calculate_route_metrics(coarsener.graph, routes, depot_id, vehicle_capacity)
    else:
        vrp = convert_graph_to_vrp_problem_inputs(graph, depot_id, vehicle_capacity)
        solver = FullQuboSolver(vrp) if solver_name == 'FullQubo' else AveragePartitionSolver(vrp)
        sol = solver.solve(qubo_params['only_one'], qubo_params['order'], qubo_params['tw_penalty'], qubo_params['backend'], qubo_params['reads'])
        formatted = []
        for r in sol.solution:
            if not r: continue
            tmp = [depot_id] + r + [depot_id]
            if len(tmp) > 2: formatted.append(tmp)
        routes = formatted if not coarsener else coarsener.inflate_route(formatted)
        metrics = calculate_route_metrics(coarsener.graph if coarsener else graph, routes, depot_id, vehicle_capacity)
    return routes, metrics


def configure_logging(level=logging.INFO):
    logging.basicConfig(level=level, format='%(levelname)s: %(message)s')
    return logging.getLogger(__name__)

logger = configure_logging()

def find_csv_files(base_dir: str) -> list:
    paths = []
    for root, _, files in os.walk(base_dir):
        for f in files:
            if f.endswith('.csv'):
                paths.append(os.path.join(root, f))
    return sorted(paths)


def log_graph_info(graph: Graph, depot_id: str, limit: int = 5):
    logger.info("\n--- Initial Graph Nodes (first %d) ---" % limit)
    for nid, node in list(graph.nodes.items())[:limit]:
        logger.info(node)
    logger.info("... and %d more nodes." % (len(graph.nodes) - limit))
    logger.info("\n--- Initial Graph Edges (first %d) ---" % limit)
    for edge in graph.edges[:limit]:
        logger.info(edge)
    logger.info(f"Total initial edges: {len(graph.edges)}")


def log_coarsening_info(coarsener: SpatioTemporalGraphCoarsener, coarsened_graph: Graph, merge_layers: list, limit: int = 5):
    logger.info("\n\n=== Coarsening Process ===")
    logger.info("--- Final Coarsened Graph Nodes (first %d) ---" % limit)
    for nid, node in list(coarsened_graph.nodes.items())[:limit]:
        logger.info(node)
    logger.info("... and %d more nodes." % (len(coarsened_graph.nodes) - limit))
    logger.info("--- Final Coarsened Graph Edges (first %d) ---" % limit)
    for edge in coarsened_graph.edges[:limit]:
        logger.info(edge)
    logger.info(f"Total final edges: {len(coarsened_graph.edges)}")
    logger.info("--- Merge Layers (first %d) ---" % limit)
    for layer in merge_layers[:limit]:
        super_id, i_id, j_id, order = layer
        logger.info(f"Super-node: {super_id} from {i_id}, {j_id} order {order}")
    logger.info(f"... and {len(merge_layers) - limit} more merge layers.")


def log_solver_results(prefix: str, routes: list, metrics: dict):
    logger.info(f"  {prefix} Routes: {routes}")
    for k, v in metrics.items():
        if isinstance(v, float):
            logger.info(f"    {k.replace('_',' ').title()}: {v:.2f}")
        else:
            logger.info(f"    {k.replace('_',' ').title()}: {v}")


def run_uncoarsened_solvers(graph: Graph, depot_id: str, capacity: float) -> dict:
    results = {}
    for name in ('Greedy', 'Savings', 'FullQubo', 'AveragePartition'):
        logger.info(f"\n--- Running UNCOARSENED {name} Solver ---")
        routes, metrics = run_solver_pipeline(graph, depot_id, capacity, name)
        key = f"Uncoarsened {name}"
        results[key] = metrics
        log_solver_results(key, routes, metrics)
    return results


def run_inflated_solvers(coarsener: SpatioTemporalGraphCoarsener, cwd_graph: Graph, depot_id: str, capacity: float) -> dict:
    results = {}
    for name in ('Greedy', 'Savings', 'FullQubo', 'AveragePartition'):
        logger.info(f"\n--- Running INFLATED {name} Solver ---")
        routes, metrics = run_solver_pipeline(cwd_graph, depot_id, capacity, name, coarsener)
        key = f"Inflated {name}"
        results[key] = metrics
        log_solver_results(key, routes, metrics)
    return results


def final_summary(all_results: dict):
    logger.info("\n\n=== FINAL SUMMARY ACROSS ALL FILES ===")
    metrics_list = [
        "total_distance", "total_service_time", "total_waiting_time",
        "total_route_duration", "total_demand_served", "time_window_violations",
        "capacity_violations", "num_vehicles", "is_feasible"
    ]
    header = "Metric".ljust(25) + " | " + " | ".join([f"{h:<20}" for h in all_results[next(iter(all_results))].keys()])
    logger.info(header)
    logger.info("-" * len(header))
    for fname, res in sorted(all_results.items()):
        logger.info(f"\n--- Results for {fname} ---")
        for m in metrics_list:
            vals = [r.get(m, 'N/A') for r in res.values()]
            formatted = [f"{v:.2f}" if isinstance(v, float) else str(v) for v in vals]
            row = m.replace('_',' ').title().ljust(25) + " | " + " | ".join([f"{fv:<20}" for fv in formatted])
            logger.info(row)
    logger.info("\nNote: 'Is Feasible' indicates if any time window or capacity violations were found.")


def process_file(csv_file_path: str) -> dict:
    logger.info(f"\n\n=== Processing file: {csv_file_path} ===")
    try:
        graph, depot_id, capacity = load_graph_from_csv(csv_file_path)
    except Exception as e:
        logger.error(f"Error loading {csv_file_path}: {e}")
        return {}
    log_graph_info(graph, depot_id)
    coarsener = SpatioTemporalGraphCoarsener(graph=graph, alpha=0.5, beta=0.5, P=0.5, radiusCoeff=1.0, depot_id=depot_id)
    coarsened_graph, merge_layers = coarsener.coarsen()
    log_coarsening_info(coarsener, coarsened_graph, merge_layers)
    uncoars = run_uncoarsened_solvers(graph, depot_id, capacity)
    inflated = run_inflated_solvers(coarsener, coarsened_graph, depot_id, capacity)
    return {**uncoars, **inflated}


def main():
    base_dir = 'solomon_dataset'
    files = find_csv_files(base_dir)
    all_results = {}
    for path in files:
        res = process_file(path)
        if res:
            all_results[os.path.basename(path)] = res
    if all_results:
        final_summary(all_results)


if __name__ == "__main__":
    main()
