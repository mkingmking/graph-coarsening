import os
import logging
from .graph import Graph, compute_euclidean_tau
from .delete2 import load_graph_from_csv, calculate_route_metrics
from .greedy_solver import GreedySolver
from .savings_solver import SavingsSolver

from .coarsener import SpatioTemporalGraphCoarsener
from .delete1 import visualize_routes






# Configure logging for better output
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration ---
# You will need to download the C101.txt file from the Solomon benchmarks
# and place it in the same directory as this script.
# The user's log shows a .csv extension, but .txt is standard for these files.

SOLOMON_INSTANCE_PATH = "/Users/MUSTAFAMERT/Desktop/graph-coarsening/graph_coarsening/C101.csv"

# Define the different parameter sets to test
# These are the hyperparameters that will be varied to see their effect on the solution
# We'll fix alpha, beta, and radiusCoeff and change only the coarsening rate 'P'
PARAMETER_SETS = [
    {"alpha": 0.5, "beta": 0.5, "P": 0.5, "radiusCoeff": 0.5, "description": ""},
    {"alpha": 0.5, "beta": 0.5, "P": 0.5, "radiusCoeff": 1.0, "description": ""},
    {"alpha": 0.5, "beta": 0.5, "P": 0.5, "radiusCoeff": 1.5, "description": ""},
    {"alpha": 0.5, "beta": 0.5, "P": 0.5, "radiusCoeff": 2.0, "description": ""},
]

def main():
    """
    Main function to load the problem, apply coarsening with different
    parameters, solve, inflate, and visualize the results.
    """
    logger.info(f"Attempting to load Solomon instance from: {SOLOMON_INSTANCE_PATH}")
    
    # Check if the file exists before proceeding
    if not os.path.exists(SOLOMON_INSTANCE_PATH):
        logger.error(f"Error: The file '{SOLOMON_INSTANCE_PATH}' was not found. Please download it and place it in the same directory.")
        return

    # Load the graph once to use as a baseline
    original_graph, depot_id, capacity = load_graph_from_csv(SOLOMON_INSTANCE_PATH)
    logger.info("Original graph loaded successfully.")

    for params in PARAMETER_SETS:
        logger.info("\n" + "="*50)
        logger.info(f"Running with parameters: {params['description']}")
        logger.info(f"  P (Target % of nodes): {params['P'] * 100}%")
        logger.info(f"  Alpha (Spatial Weight): {params['alpha']}")
        logger.info(f"  Beta (Temporal Weight): {params['beta']}")
        logger.info(f"  Radius Coefficient: {params['radiusCoeff']}")
        logger.info("="*50)

        # 1. Instantiate the coarsener with the current parameter set
        coarsener = SpatioTemporalGraphCoarsener(
            graph=original_graph,
            alpha=params['alpha'],
            beta=params['beta'],
            P=params['P'],
            radiusCoeff=params['radiusCoeff'],
            depot_id=depot_id
        )

        # 2. Perform the coarsening
        coarsened_graph, merge_layers = coarsener.coarsen()
        logger.info(f"Coarsened graph has {len(coarsened_graph.nodes)} nodes (Original: {len(original_graph.nodes)})")

        # 3. Solve on the coarsened graph using the Greedy Solver
        #solver = GreedySolver(graph=coarsened_graph, depot_id=depot_id, vehicle_capacity=capacity)
        solver = SavingsSolver(graph=coarsened_graph, depot_id=depot_id, vehicle_capacity=capacity)

        coarsened_routes, _ = solver.solve()
        
        # 4. Inflate the solution back to the original graph
        inflated_routes = coarsener.inflate_route(coarsened_routes)
        
        # 5. Calculate final metrics on the inflated routes
        final_metrics = calculate_route_metrics(original_graph, inflated_routes, depot_id, capacity)
        
        # Defensive check to handle cases where no routes were found
        if not final_metrics:
            logger.error("Failed to calculate final metrics.")
            continue # Skip to the next parameter set
            
        logger.info(f"Final Metrics for this run:")
        logger.info(f"  Total Distance: {final_metrics['total_distance']:.2f}")
        logger.info(f"  Number of Vehicles: {final_metrics['num_vehicles']}")
        logger.info(f"  Feasibility: {'Yes' if final_metrics['feasible'] else 'No'} (Time: {final_metrics['tw_violations_count']}, Capacity: {final_metrics['cap_violations_count']})")
        logger.info("-"*50)

        # 6. Visualize the final solution, passing the correct depot ID
        visualize_routes(original_graph, inflated_routes, f"Coarsened Solution: {params['description']}", depot_id)

if __name__ == "__main__":
    main()
