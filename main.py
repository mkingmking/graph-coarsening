import os
import logging
import itertools # For grid search (though not directly used in this refactored main, kept for context)
import random # For random search (though not directly used in this refactored main, kept for context)

# Configure logging
# Set to INFO to see detailed progress messages.
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Import classes and functions from separate files
from graph import Graph, compute_euclidean_tau
from utils import load_graph_from_csv, calculate_route_metrics

# Import the traditional solvers
from greedy_solver import GreedySolver
from savings_solver import SavingsSolver

# Import the coarsener
from coarsener import SpatioTemporalGraphCoarsener

# Import the new VRP/QUBO related classes
from quantum_solvers.vrp_problem_qubo import VRPProblem
from quantum_solvers.vrp_solution import VRPSolution
from quantum_solvers.vrp_solvers import FullQuboSolver, AveragePartitionSolver

# IMPORTANT: Replace this mock with your actual DWaveSolvers_modified import
# from DWaveSolvers_modified import solve_qubo
# For local testing, we use the mock:
from quantum_solvers.mock_dwave_solvers import MockDWaveSolvers as DWaveSolvers_modified


def convert_graph_to_vrp_problem_inputs(graph: Graph, depot_id: str, vehicle_capacity: float):
    """
    Converts a Graph object (original or coarsened) into the input format
    expected by the VRPProblem class for QUBO formulation.

    Args:
        graph (Graph): The graph (original or coarsened).
        depot_id (str): The ID of the depot node.
        vehicle_capacity (float): The capacity of each vehicle.

    Returns:
        VRPProblem: An instance of VRPProblem configured with the graph data.
    """
    # Create cost and time_cost matrices as dictionaries of dictionaries
    # This allows using original node IDs (strings) as keys directly.
    costs_matrix = {u_id: {} for u_id in graph.nodes.keys()}
    time_costs_matrix = {u_id: {} for u_id in graph.nodes.keys()}

    for u_id, u_node in graph.nodes.items():
        for v_id, v_node in graph.nodes.items():
            if u_id == v_id:
                costs_matrix[u_id][v_id] = 0.0
                time_costs_matrix[u_id][v_id] = 0.0
            else:
                tau = compute_euclidean_tau(u_node, v_node)
                costs_matrix[u_id][v_id] = tau
                time_costs_matrix[u_id][v_id] = tau # For VRPProblem, time_costs are travel times

    # Prepare VRPProblem specific inputs
    # For the QUBO, we are solving for a fixed number of vehicles.
    # This `num_vehicles_for_qubo` will be the length of the `capacities` list passed to VRPProblem.
    # Let's assume we want to use 3 vehicles for the QUBO part. This can be tuned.
    num_vehicles_for_qubo = 3 # This can be a parameter in your tuning script too!
    capacities_for_qubo = [vehicle_capacity] * num_vehicles_for_qubo

    customer_ids = [node_id for node_id in graph.nodes if node_id != depot_id]
    customer_demands = {node_id: graph.nodes[node_id].demand for node_id in customer_ids}

    # Return the VRPProblem instance
    return VRPProblem(graph.nodes, depot_id, costs_matrix, time_costs_matrix, capacities_for_qubo, customer_ids, customer_demands)


def run_solver_pipeline(graph: Graph, depot_id: str, vehicle_capacity: float, solver_name: str, coarsener_instance: SpatioTemporalGraphCoarsener = None):
    """
    Runs a specified solver on the given graph (either original or coarsened),
    and if a coarsener is provided, inflates the routes.
    Calculates and returns the metrics.

    Args:
        graph (Graph): The graph to solve on (can be initial or coarsened).
        depot_id (str): The ID of the depot node.
        vehicle_capacity (float): The capacity of each vehicle.
        solver_name (str): Name of the solver to use ('Greedy', 'Savings', 'FullQubo', 'AveragePartition').
        coarsener_instance (SpatioTemporalGraphCoarsener, optional): The coarsener instance
                                                                      if inflation is needed. Defaults to None.

    Returns:
        tuple: (routes, metrics_dict)
               routes (list): The generated or inflated routes.
               metrics_dict (dict): A dictionary of detailed route metrics.
    """
    routes = []
    metrics = {}
    
    # Fixed QUBO parameters for this demonstration (these would be tuned in run_tuning.py)
    qubo_only_one_const = 1000
    qubo_order_const = 1
    qubo_tw_penalty_const = 1000
    qubo_num_reads = 10
    qubo_solver_type_backend = 'simulated' # Using the mock solver

    if solver_name in ['Greedy', 'Savings']:
        if solver_name == 'Greedy':
            solver = GreedySolver(graph, depot_id, vehicle_capacity)
        else: # Savings
            solver = SavingsSolver(graph, depot_id, vehicle_capacity)
        
        routes, metrics = solver.solve() # These solvers return (routes, metrics)
        
        # If coarsener_instance is provided, it means this is a run on a coarsened graph
        # and the routes need to be inflated.
        if coarsener_instance:
            logger.info(f"  Inflating routes from {solver_name} solver...")
            # Ensure routes are properly formatted with depot at start/end for inflation
            formatted_coarsened_routes = []
            for route in routes:
                if not route: # Skip empty routes
                    continue
                temp_route = list(route) # Make a copy
                if temp_route[0] != depot_id:
                    temp_route.insert(0, depot_id)
                if temp_route[-1] != depot_id:
                    temp_route.append(depot_id)
                
                # Only add if it has more than just the depot (i.e., at least one customer)
                if len(temp_route) > 2:
                    formatted_coarsened_routes.append(temp_route)
            
            routes = coarsener_instance.inflate_route(formatted_coarsened_routes)
            # Recalculate metrics on the inflated routes on the ORIGINAL graph
            metrics = calculate_route_metrics(coarsener_instance.graph, routes, depot_id, vehicle_capacity)

    elif solver_name in ['FullQubo', 'AveragePartition']:
        # Convert the current graph (coarsened or uncoarsened) to VRPProblem format
        vrp_problem = convert_graph_to_vrp_problem_inputs(graph, depot_id, vehicle_capacity)

        if solver_name == 'FullQubo':
            solver = FullQuboSolver(vrp_problem)
        else: # AveragePartition
            solver = AveragePartitionSolver(vrp_problem)
        
        # Solve the QUBO problem
        qubo_vrp_solution = solver.solve(qubo_only_one_const, qubo_order_const, qubo_tw_penalty_const, qubo_solver_type_backend, qubo_num_reads)
        
        # The solution from QUBO is based on string IDs of (coarsened) nodes.
        # The `solution` attribute of VRPSolution already contains string IDs.
        routes_from_qubo = qubo_vrp_solution.solution
        
        # Ensure routes are properly formatted with depot at start/end for inflation
        formatted_routes_for_inflation = []
        for route in routes_from_qubo:
            if not route: # Skip empty routes
                continue
            temp_route = list(route) # Make a copy
            if temp_route[0] != depot_id:
                temp_route.insert(0, depot_id)
            if temp_route[-1] != depot_id:
                temp_route.append(depot_id)
            
            # Only add if it has more than just the depot (i.e., at least one customer)
            if len(temp_route) > 2:
                formatted_routes_for_inflation.append(temp_route)
        
        routes = formatted_routes_for_inflation # These are the routes on the (coarsened) graph

        if coarsener_instance:
            logger.info(f"  Inflating routes from {solver_name} solver...")
            routes = coarsener_instance.inflate_route(routes)
            # Recalculate metrics on the inflated routes on the ORIGINAL graph
            metrics = calculate_route_metrics(coarsener_instance.graph, routes, depot_id, vehicle_capacity)
        else:
            # If no coarsener_instance, it means we ran on the initial graph.
            # Calculate metrics directly on the routes obtained from QUBO solver on initial graph.
            metrics = calculate_route_metrics(graph, routes, depot_id, vehicle_capacity)

    else:
        raise ValueError(f"Unknown solver name: {solver_name}")

    return routes, metrics


def process_file(csv_file_path: str) -> dict:
    """
    Processes a single Solomon dataset file:
    1. Loads the graph.
    2. Runs Greedy and Savings solvers on the uncoarsened graph.
    3. Runs coarsening.
    4. Runs Greedy and Savings solvers on the coarsened graph and inflates.
    5. Runs FullQubo and AveragePartition solvers on the uncoarsened graph.
    6. Runs FullQubo and AveragePartition solvers on the coarsened graph and inflates.
    7. Collects and returns all metrics.
    """
    file_name_only = os.path.basename(csv_file_path)
    current_file_results = {}

    logger.info(f"\n\n=====================================================================================")
    logger.info(f"Processing file: {csv_file_path}")
    logger.info(f"=====================================================================================")

    # --- Load graph from CSV ---
    try:
        initial_graph, depot_id, VEHICLE_CAPACITY = load_graph_from_csv(csv_file_path)
    except Exception as e:
        logger.error(f"Skipping {csv_file_path} due to error loading graph: {e}")
        return {} # Return empty results if loading fails

    logger.info("\n--- Initial Graph Nodes (from CSV) ---")
    for node_id, node in list(initial_graph.nodes.items())[:5]: # Print first 5 nodes
        logger.info(node)
    logger.info(f"... and {len(initial_graph.nodes) - 5} more nodes.")
    logger.info("\n--- Initial Graph Edges (from CSV, first 5) ---")
    for i, edge in enumerate(initial_graph.edges):
        if i >= 5: break
        logger.info(edge)
    logger.info(f"Total initial edges: {len(initial_graph.edges)}")

    # --- Coarsening Process (fixed parameters for this demo) ---
    logger.info("\n\n=== Starting Coarsening Process ===")
    coarsener = SpatioTemporalGraphCoarsener(
        graph=initial_graph,
        alpha=0.5,
        beta=0.5,
        P=0.5, # Reduce to 50% of original nodes
        radiusCoeff=1.0,
        depot_id=depot_id
    )
    coarsened_graph, merge_layers = coarsener.coarsen()

    logger.info("\n--- Final Coarsened Graph Nodes ---")
    for node_id, node in list(coarsened_graph.nodes.items())[:5]: # Print first 5 nodes
        logger.info(node)
    logger.info(f"... and {len(coarsened_graph.nodes) - 5} more nodes.")
    logger.info("\n--- Final Coarsened Graph Edges (first 5) ---")
    for i, edge in enumerate(coarsened_graph.edges):
        if i >= 5: break
        logger.info(edge)
    logger.info(f"Total final edges: {len(coarsened_graph.edges)}")

    logger.info("\n--- Merge Layers (for Inflation) ---")
    for layer in merge_layers[:5]: # Print first 5 merge layers
        super_node_id, node_i_id, node_j_id, pi_order = layer
        logger.info(f"Super-node: {super_node_id} formed from {node_i_id} and {node_j_id} in order {pi_order}")
    logger.info(f"... and {len(merge_layers) - 5} more merge layers.")

    logger.info("\n--- Original nodes represented by final super-nodes (first 5) ---")
    count_super_nodes_printed = 0
    for node_id, node in coarsened_graph.nodes.items():
        if node.is_super_node:
            logger.info(f"Super-node {node_id} represents original nodes: {node.original_nodes}")
            count_super_nodes_printed += 1
            if count_super_nodes_printed >= 5:
                break
    logger.info(f"... and more super-nodes.")


    # --- Run Solvers on UNCOARSENED Graph ---
    logger.info("\n\n=== Solving on UNCOARSENED Graph ===")
    
    # Greedy Solver
    logger.info("\n--- Running UNCOARSENED Greedy Solver ---")
    uncoarsened_greedy_routes, uncoarsened_greedy_metrics = run_solver_pipeline(initial_graph, depot_id, VEHICLE_CAPACITY, 'Greedy')
    current_file_results['Uncoarsened Greedy'] = uncoarsened_greedy_metrics
    logger.info(f"  Routes: {uncoarsened_greedy_routes}")
    for key, value in uncoarsened_greedy_metrics.items():
        if isinstance(value, float): logger.info(f"    {key.replace('_', ' ').title()}: {value:.2f}")
        else: logger.info(f"    {key.replace('_', ' ').title()}: {value}")

    # Savings Solver
    logger.info("\n--- Running UNCOARSENED Savings Solver ---")
    uncoarsened_savings_routes, uncoarsened_savings_metrics = run_solver_pipeline(initial_graph, depot_id, VEHICLE_CAPACITY, 'Savings')
    current_file_results['Uncoarsened Savings'] = uncoarsened_savings_metrics
    logger.info(f"  Routes: {uncoarsened_savings_routes}")
    for key, value in uncoarsened_savings_metrics.items():
        if isinstance(value, float): logger.info(f"    {key.replace('_', ' ').title()}: {value:.2f}")
        else: logger.info(f"    {key.replace('_', ' ').title()}: {value}")

    # FullQubo Solver
    logger.info("\n--- Running UNCOARSENED FullQubo Solver ---")
    uncoarsened_fullqubo_routes, uncoarsened_fullqubo_metrics = run_solver_pipeline(initial_graph, depot_id, VEHICLE_CAPACITY, 'FullQubo')
    current_file_results['Uncoarsened FullQubo'] = uncoarsened_fullqubo_metrics
    logger.info(f"  Routes: {uncoarsened_fullqubo_routes}")
    for key, value in uncoarsened_fullqubo_metrics.items():
        if isinstance(value, float): logger.info(f"    {key.replace('_', ' ').title()}: {value:.2f}")
        else: logger.info(f"    {key.replace('_', ' ').title()}: {value}")

    # AveragePartition Solver
    logger.info("\n--- Running UNCOARSENED AveragePartition Solver ---")
    uncoarsened_aps_routes, uncoarsened_aps_metrics = run_solver_pipeline(initial_graph, depot_id, VEHICLE_CAPACITY, 'AveragePartition')
    current_file_results['Uncoarsened AveragePartition'] = uncoarsened_aps_metrics
    logger.info(f"  Routes: {uncoarsened_aps_routes}")
    for key, value in uncoarsened_aps_metrics.items():
        if isinstance(value, float): logger.info(f"    {key.replace('_', ' ').title()}: {value:.2f}")
        else: logger.info(f"    {key.replace('_', ' ').title()}: {value}")


    # --- Run Solvers on COARSENED Graph and Inflate ---
    logger.info("\n\n=== Solving on COARSENED Graph and Inflating ===")

    # Greedy Solver
    logger.info("\n--- Running INFLATED Greedy Solver ---")
    inflated_greedy_routes, inflated_greedy_metrics = run_solver_pipeline(coarsened_graph, depot_id, VEHICLE_CAPACITY, 'Greedy', coarsener_instance=coarsener)
    current_file_results['Inflated Greedy'] = inflated_greedy_metrics
    logger.info(f"  Routes: {inflated_greedy_routes}")
    for key, value in inflated_greedy_metrics.items():
        if isinstance(value, float): logger.info(f"    {key.replace('_', ' ').title()}: {value:.2f}")
        else: logger.info(f"    {key.replace('_', ' ').title()}: {value}")

    # Savings Solver
    logger.info("\n--- Running INFLATED Savings Solver ---")
    inflated_savings_routes, inflated_savings_metrics = run_solver_pipeline(coarsened_graph, depot_id, VEHICLE_CAPACITY, 'Savings', coarsener_instance=coarsener)
    current_file_results['Inflated Savings'] = inflated_savings_metrics
    logger.info(f"  Routes: {inflated_savings_routes}")
    for key, value in inflated_savings_metrics.items():
        if isinstance(value, float): logger.info(f"    {key.replace('_', ' ').title()}: {value:.2f}")
        else: logger.info(f"    {key.replace('_', ' ').title()}: {value}")

    # FullQubo Solver
    logger.info("\n--- Running INFLATED FullQubo Solver ---")
    inflated_fullqubo_routes, inflated_fullqubo_metrics = run_solver_pipeline(coarsened_graph, depot_id, VEHICLE_CAPACITY, 'FullQubo', coarsener_instance=coarsener)
    current_file_results['Inflated FullQubo'] = inflated_fullqubo_metrics
    logger.info(f"  Routes: {inflated_fullqubo_routes}")
    for key, value in inflated_fullqubo_metrics.items():
        if isinstance(value, float): logger.info(f"    {key.replace('_', ' ').title()}: {value:.2f}")
        else: logger.info(f"    {key.replace('_', ' ').title()}: {value}")

    # AveragePartition Solver
    logger.info("\n--- Running INFLATED AveragePartition Solver ---")
    inflated_aps_routes, inflated_aps_metrics = run_solver_pipeline(coarsened_graph, depot_id, VEHICLE_CAPACITY, 'AveragePartition', coarsener_instance=coarsener)
    current_file_results['Inflated AveragePartition'] = inflated_aps_metrics
    logger.info(f"  Routes: {inflated_aps_routes}")
    for key, value in inflated_aps_metrics.items():
        if isinstance(value, float): logger.info(f"    {key.replace('_', ' ').title()}: {value:.2f}")
        else: logger.info(f"    {key.replace('_', ' ').title()}: {value}")


    return current_file_results


if __name__ == "__main__":
    base_dataset_dir = 'solomon_dataset' 
    all_csv_file_paths = []

    for root, dirs, files in os.walk(base_dataset_dir):
        for file in files:
            if file.endswith('.csv'):
                full_path = os.path.join(root, file)
                all_csv_file_paths.append(full_path)
    
    all_csv_file_paths.sort()

    all_results = {}

    for csv_file_path in all_csv_file_paths:
        file_name_only = os.path.basename(csv_file_path)
        file_results = process_file(csv_file_path)
        if file_results: # Only store if processing was successful
            all_results[file_name_only] = file_results

    # --- Final Summary Comparison Across All Files ---
    logger.info("\n\n=====================================================================================")
    logger.info("======================== FINAL SUMMARY ACROSS ALL FILES =============================")
    logger.info("=====================================================================================")

    metrics_to_compare = [
        "total_distance",
        "total_service_time",
        "total_waiting_time",
        "total_route_duration",
        "total_demand_served",
        "time_window_violations",
        "capacity_violations",
        "num_vehicles",
        "is_feasible"
    ]

    # Print headers for the summary table
    logger.info(f"{'Metric':<25} | {'Uncoarsened Greedy':<20} | {'Uncoarsened Savings':<20} | {'Uncoarsened FullQubo':<20} | {'Uncoarsened AvgPart':<20} | {'Inflated Greedy':<20} | {'Inflated Savings':<20} | {'Inflated FullQubo':<20} | {'Inflated AvgPart':<20}")
    logger.info("-" * 200) # Extend separator line

    for file_name in sorted(all_results.keys()): # Iterate in sorted order of file names
        results = all_results[file_name]
        logger.info(f"\n--- Results for {file_name} ---")
        
        for metric in metrics_to_compare:
            uncoarsened_greedy_val = results.get('Uncoarsened Greedy', {}).get(metric, 'N/A')
            uncoarsened_savings_val = results.get('Uncoarsened Savings', {}).get(metric, 'N/A')
            uncoarsened_fullqubo_val = results.get('Uncoarsened FullQubo', {}).get(metric, 'N/A')
            uncoarsened_aps_val = results.get('Uncoarsened AveragePartition', {}).get(metric, 'N/A')
            inflated_greedy_val = results.get('Inflated Greedy', {}).get(metric, 'N/A')
            inflated_savings_val = results.get('Inflated Savings', {}).get(metric, 'N/A')
            inflated_fullqubo_val = results.get('Inflated FullQubo', {}).get(metric, 'N/A')
            inflated_aps_val = results.get('Inflated AveragePartition', {}).get(metric, 'N/A')

            def format_val(val):
                if isinstance(val, float):
                    return f"{val:.2f}"
                return str(val)

            logger.info(f"{metric.replace('_', ' ').title():<25} | {format_val(uncoarsened_greedy_val):<20} | {format_val(uncoarsened_savings_val):<20} | {format_val(uncoarsened_fullqubo_val):<20} | {format_val(uncoarsened_aps_val):<20} | {format_val(inflated_greedy_val):<20} | {format_val(inflated_savings_val):<20} | {format_val(inflated_fullqubo_val):<20} | {format_val(inflated_aps_val):<20}")

    logger.info("\nNote: 'Is Feasible' indicates if any time window or capacity violations were found across all routes.")
    logger.info("A well-functioning coarsening/inflation should ideally result in feasible inflated routes serving all demand.")

