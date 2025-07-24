import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Import classes and functions from separate files
from graph import Graph
from utils import load_graph_from_csv, calculate_route_metrics
from coarsener import SpatioTemporalGraphCoarsener
from greedy_solver import GreedySolver
from savings_solver import SavingsSolver

if __name__ == "__main__":
    # Define the base directory where 'solomon_dataset' is located
    base_dataset_dir = 'solomon_dataset' 

    # List to store full paths of all CSV files found
    all_csv_file_paths = []

    # Walk through the base_dataset_dir to find all CSV files
    for root, dirs, files in os.walk(base_dataset_dir):
        for file in files:
            if file.endswith('.csv'):
                full_path = os.path.join(root, file)
                all_csv_file_paths.append(full_path)
    
    # Sort the file paths for consistent processing order (e.g., C101, C102, ..., R101, etc.)
    all_csv_file_paths.sort()

    # Store results for all files
    all_results = {}

    for csv_file_path in all_csv_file_paths:
        # Extract just the file name for display in results
        file_name_only = os.path.basename(csv_file_path)

        logger.info(f"\n\n=====================================================================================")
        logger.info(f"Processing file: {csv_file_path}")
        logger.info(f"=====================================================================================")

        # --- Load graph from CSV ---
        try:
            initial_graph, depot_id, VEHICLE_CAPACITY = load_graph_from_csv(csv_file_path)
        except Exception as e:
            logger.error(f"Skipping {csv_file_path} due to error loading graph: {e}")
            continue # Skip to the next file if loading fails

        current_file_results = {}

        logger.info("\n--- Initial Graph Nodes (from CSV) ---")
        for node_id, node in list(initial_graph.nodes.items())[:5]: # Print first 5 nodes
            logger.info(node)
        logger.info(f"... and {len(initial_graph.nodes) - 5} more nodes.")
        logger.info("\n--- Initial Graph Edges (from CSV, first 5) ---")
        for i, edge in enumerate(initial_graph.edges):
            if i >= 5: break
            logger.info(edge)
        logger.info(f"Total initial edges: {len(initial_graph.edges)}")

        # --- Solve directly on the uncoarsened graph with Greedy Solver ---
        logger.info("\n\n=== Solving on UNCOARSENED Graph (Greedy Solver) ===")
        uncoarsened_greedy_solver = GreedySolver(initial_graph, depot_id, VEHICLE_CAPACITY)
        uncoarsened_greedy_routes, _ = uncoarsened_greedy_solver.solve() # Metrics are calculated here
        uncoarsened_greedy_metrics = calculate_route_metrics(initial_graph, uncoarsened_greedy_routes, depot_id, VEHICLE_CAPACITY)
        current_file_results['Uncoarsened Greedy'] = uncoarsened_greedy_metrics
        
        logger.info("\n--- Metrics for UNCOARSENED Greedy Routes ---")
        logger.info(f"Routes: {uncoarsened_greedy_routes}")
        for key, value in uncoarsened_greedy_metrics.items():
            if isinstance(value, float):
                logger.info(f"{key.replace('_', ' ').title()}: {value:.2f}")
            else:
                logger.info(f"{key.replace('_', ' ').title()}: {value}")


        # --- Solve directly on the uncoarsened graph with Savings Solver ---
        logger.info("\n\n=== Solving on UNCOARSENED Graph (Savings Solver) ===")
        uncoarsened_savings_solver = SavingsSolver(initial_graph, depot_id, VEHICLE_CAPACITY)
        uncoarsened_savings_routes, _ = uncoarsened_savings_solver.solve() # Metrics are calculated here
        uncoarsened_savings_metrics = calculate_route_metrics(initial_graph, uncoarsened_savings_routes, depot_id, VEHICLE_CAPACITY)
        current_file_results['Uncoarsened Savings'] = uncoarsened_savings_metrics
        
        logger.info("\n--- Metrics for UNCOARSENED Savings Routes ---")
        logger.info(f"Routes: {uncoarsened_savings_routes}")
        for key, value in uncoarsened_savings_metrics.items():
            if isinstance(value, float):
                logger.info(f"{key.replace('_', ' ').title()}: {value:.2f}")
            else:
                logger.info(f"{key.replace('_', ' ').title()}: {value}")


        # --- Coarsening Process ---
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


        # --- Solve on the coarsened graph with Greedy Solver and then inflate ---
        logger.info("\n\n=== Solving on COARSENED Graph (Greedy Solver) and Inflating ===")
        coarsened_greedy_solver = GreedySolver(coarsened_graph, depot_id, VEHICLE_CAPACITY)
        coarsened_greedy_routes, _ = coarsened_greedy_solver.solve() # Metrics are calculated here
        
        final_inflated_greedy_routes = coarsener.inflate_route(coarsened_greedy_routes)

        # --- Calculate metrics for the inflated Greedy route ---
        inflated_greedy_metrics = calculate_route_metrics(initial_graph, final_inflated_greedy_routes, depot_id, VEHICLE_CAPACITY)
        current_file_results['Inflated Greedy'] = inflated_greedy_metrics

        logger.info("\n--- Metrics for INFLATED Greedy Routes (on Original Graph) ---")
        logger.info(f"Routes: {final_inflated_greedy_routes}")
        for key, value in inflated_greedy_metrics.items():
            if isinstance(value, float):
                logger.info(f"{key.replace('_', ' ').title()}: {value:.2f}")
            else:
                logger.info(f"{key.replace('_', ' ').title()}: {value}")

        # --- Solve on the coarsened graph with Savings Solver and then inflate ---
        logger.info("\n\n=== Solving on COARSENED Graph (Savings Solver) and Inflating ===")
        coarsened_savings_solver = SavingsSolver(coarsened_graph, depot_id, VEHICLE_CAPACITY)
        coarsened_savings_routes, _ = coarsened_savings_solver.solve() # Metrics are calculated here
        
        final_inflated_savings_routes = coarsener.inflate_route(coarsened_savings_routes)

        # --- Calculate metrics for the inflated Savings route ---
        inflated_savings_metrics = calculate_route_metrics(initial_graph, final_inflated_savings_routes, depot_id, VEHICLE_CAPACITY)
        current_file_results['Inflated Savings'] = inflated_savings_metrics

        logger.info("\n--- Metrics for INFLATED Savings Routes (on Original Graph) ---")
        logger.info(f"Routes: {final_inflated_savings_routes}")
        for key, value in inflated_savings_metrics.items():
            if isinstance(value, float):
                logger.info(f"{key.replace('_', ' ').title()}: {value:.2f}")
            else:
                logger.info(f"{key.replace('_', ' ').title()}: {value}")

        # Store results for the current file
        all_results[file_name_only] = current_file_results # Store by just the file name

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
    print(f"{'Metric':<25} | {'Uncoarsened Greedy':<20} | {'Uncoarsened Savings':<20} | {'Inflated Greedy':<20} | {'Inflated Savings':<20}")
    print("-" * 120)

    for file_name in sorted(all_results.keys()): # Iterate in sorted order of file names
        results = all_results[file_name]
        print(f"\n--- Results for {file_name} ---")
        
        for metric in metrics_to_compare:
            uncoarsened_greedy_val = results.get('Uncoarsened Greedy', {}).get(metric, 'N/A')
            uncoarsened_savings_val = results.get('Uncoarsened Savings', {}).get(metric, 'N/A')
            inflated_greedy_val = results.get('Inflated Greedy', {}).get(metric, 'N/A')
            inflated_savings_val = results.get('Inflated Savings', {}).get(metric, 'N/A')

            def format_val(val):
                if isinstance(val, float):
                    return f"{val:.2f}"
                return str(val)

            print(f"{metric.replace('_', ' ').title():<25} | {format_val(uncoarsened_greedy_val):<20} | {format_val(uncoarsened_savings_val):<20} | {format_val(inflated_greedy_val):<20} | {format_val(inflated_savings_val):<20}")

    print("\nNote: 'Is Feasible' indicates if any time window or capacity violations were found across all routes.")
    print("A well-functioning coarsening/inflation should ideally result in feasible inflated routes serving all demand.")
