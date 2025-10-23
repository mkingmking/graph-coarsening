import os
import logging
import json
import time
import argparse
from pathlib import Path

from .graph import Graph, compute_euclidean_tau
from .utils import load_graph_from_csv, calculate_route_metrics
from .coarsener import SpatioTemporalGraphCoarsener
from .quantum_solvers.vrp_problem import VRPProblem
from .quantum_solvers.vrp_solvers import FullQuboSolver, AveragePartitionSolver
from .visualisation import visualize_routes

# --- Visualization Counters ---
_visualisation_counter_uncoarsened_quantum = {}
_visualisation_counter_coarsened_quantum = {}


# --- Helper Functions for Data Conversion ---

def convert_graph_to_vrp_problem_inputs(graph: Graph, depot_id: str, vehicle_capacity: float) -> tuple[VRPProblem, list]:
    """
    Converts the project's Graph object into the format required by VRPProblem,
    using integer indices for nodes.
    """
    # Sorting handles both numeric and string-based super-node IDs
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

    num_vehicles = len(customer_ids)
    capacities = [vehicle_capacity] * num_vehicles
    customer_ints = [id_to_int_map[nid] for nid in customer_ids]

    vrp_problem = VRPProblem(
        source_depot=int_depot_id, costs=costs, time_costs=time_costs,
        capacities=capacities, dests=customer_ints, weights=demands,
        time_windows=time_windows, service_times=service_times
    )
    return vrp_problem, int_to_id_map

def map_solution_to_original_ids(solution_routes_int: list, int_to_id_map: list) -> list:
    """Maps routes with integer IDs back to original string IDs."""
    mapped_routes = []
    for route_int in solution_routes_int:
        if route_int:
            mapped_routes.append([int_to_id_map[i] for i in route_int])
    return mapped_routes

# --- Main Solver Pipeline ---

def run_solver_pipeline(graph: Graph, depot_id: str, vehicle_capacity: float, solver_name: str, coarsener: SpatioTemporalGraphCoarsener = None):
    start_time = time.perf_counter()
    
    # Tuned parameters for quantum solvers
    qubo_params = {
        'only_one': 2000000, 'capacity_penalty': 100000,
        'time_window_penalty': 10000, 'vehicle_start_cost': 5000,
        'order': 1, 'backend': 'simulated', 'reads': 2000
    }

    vrp, int_to_id_map = convert_graph_to_vrp_problem_inputs(graph, depot_id, vehicle_capacity)
    
    if solver_name == 'FullQubo':
        solver = FullQuboSolver(vrp)
    elif solver_name == 'AveragePartitionSolver':
        solver = AveragePartitionSolver(vrp)
    else:
        raise ValueError(f"Unknown solver: {solver_name}")

    sol = solver.solve(
        qubo_params['only_one'], qubo_params['order'], qubo_params['capacity_penalty'],
        qubo_params['time_window_penalty'], qubo_params['vehicle_start_cost'],
        qubo_params['backend'], qubo_params['reads']
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
        metrics_graph = coarsener.graph # Use original graph for final metrics

    metrics = calculate_route_metrics(metrics_graph, routes, depot_id, vehicle_capacity)
    
    end_time = time.perf_counter()
    duration = end_time - start_time
    return routes, metrics, duration

# --- Logging and Reporting ---

def configure_logging(level=logging.INFO):
    logging.basicConfig(level=level, format='%(levelname)s: %(message)s')
    return logging.getLogger(__name__)

logger = configure_logging()

def log_solver_results(prefix: str, routes: list, metrics: dict, duration: float):
    logger.info(f"\n--- {prefix} Results ---")
    logger.info(f"  Computation Time: {duration:.4f} seconds")
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

# --- Main Execution Flow ---

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

def process_file(csv_file_path: str, num_customers: int) -> dict:
    logger.info(f"\n\n=== Processing file: {Path(csv_file_path).name} with {num_customers} customers ===")
    try:
        full_graph, depot_id, capacity = load_graph_from_csv(csv_file_path)
    except Exception as e:
        logger.error(f"Error loading {csv_file_path}: {e}")
        return {}

    subgraph = create_subgraph(full_graph, depot_id, num_customers)
    
    file_results = {}
    solvers_to_run = ('FullQubo', 'AveragePartitionSolver')
    
    #Define an absolute path for the visualization save directory
    script_dir = Path(__file__).resolve().parent
    save_dir = script_dir / "quantum_visualisations"
    save_dir.mkdir(exist_ok=True)

    # Run UNCOARSENED solvers
    for name in solvers_to_run:
        routes, metrics, duration = run_solver_pipeline(subgraph, depot_id, capacity, name)
        metrics['computation_time'] = duration
        file_results[f"Uncoarsened {name}"] = metrics
        log_solver_results(f"Uncoarsened {name}", routes, metrics, duration)
        
        # Visualize uncoarsened solution
        base_filename = Path(csv_file_path).stem
        count = _visualisation_counter_uncoarsened_quantum.get(name, 0) + 1
        _visualisation_counter_uncoarsened_quantum[name] = count
        filename = f"{base_filename}_{name}_uncoarsened_{count}.png"
        absolute_filepath = save_dir / filename
        visualize_routes(
            subgraph, routes, depot_id, 
            title=f"{base_filename} Uncoarsened - {name}", 
            filename=str(absolute_filepath)
        )

    # Run COARSENED solvers
    coarsener = SpatioTemporalGraphCoarsener(graph=subgraph, alpha=0.8, beta=0.4, P=0.5, radiusCoeff=2.0, depot_id=depot_id)
    coarsened_graph, _ = coarsener.coarsen()
    for name in solvers_to_run:
        routes, metrics, duration = run_solver_pipeline(coarsened_graph, depot_id, capacity, name, coarsener)
        metrics['computation_time'] = duration
        file_results[f"Inflated {name}"] = metrics
        log_solver_results(f"Inflated {name}", routes, metrics, duration)

        # Visualize coarsened (inflated) solution 
        base_filename = Path(csv_file_path).stem
        count = _visualisation_counter_coarsened_quantum.get(name, 0) + 1
        _visualisation_counter_coarsened_quantum[name] = count
        filename = f"{base_filename}_{name}_coarsened_{count}.png"
        absolute_filepath = save_dir / filename
        visualize_routes(
            subgraph, routes, depot_id, 
            title=f"{base_filename} Coarsened - {name}", 
            filename=str(absolute_filepath)
        )
        
    return file_results

def main():
    parser = argparse.ArgumentParser(description="Run Quantum VRP Solvers with and without Graph Coarsening.")
    parser.add_argument("--file", type=str, default=None, help="Path to a single Solomon CSV file to process.")
    parser.add_argument("--data", type=str, default=None, help="Directory containing Solomon CSV files.")
    parser.add_argument("--customers", type=int, default=10, help="Number of customers to include in the sub-problem.")
    parser.add_argument("--output", type=str, help="Path to a JSON file to save the detailed results.")
    args = parser.parse_args()

    if args.file:
        files_to_process = [args.file]
        if not Path(args.file).is_file():
            logger.error(f"File not found: {args.file}")
            return
    else:
        script_dir = Path(__file__).resolve().parent
        data_dir = Path(args.data) if args.data else script_dir / "solomon_dataset"
        if not data_dir.exists():
            logger.error(f"Data directory not found: {data_dir}. Use --data or place data in a 'solomon_dataset' folder.")
            return
        files_to_process = [str(p) for p in sorted(data_dir.rglob("*.csv"))]
        if not files_to_process:
            logger.warning(f"No CSV files found in {data_dir}.")
            return

    all_results = {}
    for csv_path in files_to_process:
        results = process_file(csv_path, args.customers)
        all_results[csv_path] = results
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(all_results, f, indent=4)
        logger.info(f"\nResults for {len(files_to_process)} file(s) saved to {args.output}")

    final_summary(all_results)
    logger.info("\nAll done.")

if __name__ == "__main__":
    main()

