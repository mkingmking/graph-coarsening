import os
import logging
import json
import itertools

from graph import Graph
from utils import load_graph_from_csv, calculate_route_metrics
from greedy_solver import GreedySolver
from coarsener import SpatioTemporalGraphCoarsener
from utils import compute_euclidean_tau

# Set up logging for better visibility
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Mock D-Wave solver for local testing - this is not used in the new script
# from quantum_solvers.mock_dwave_solvers import MockDWaveSolvers as DWaveSolvers_modified
# from visualisation import visualize_routes
# from or_tools_solver import ORToolsSolver

def run_experiment(
    csv_file_path: str,
    alpha: float,
    beta: float,
    P: float,
    radiusCoeff: float
) -> dict:
    """
    Runs a single experiment with a specific set of hyperparameters.
    """
    try:
        graph, depot_id, capacity = load_graph_from_csv(csv_file_path)
    except Exception as e:
        logger.error(f"Error loading {csv_file_path}: {e}")
        return {}

    logger.info(f"\n--- Running experiment with alpha={alpha}, beta={beta}, P={P}, radiusCoeff={radiusCoeff} ---")

    # Run uncoarsened greedy solver
    greedy_solver = GreedySolver(graph=graph, depot_id=depot_id, vehicle_capacity=capacity)
    uncoarsened_routes, uncoarsened_metrics = greedy_solver.solve()
    
    # Run the coarsening and inflation process
    coarsener = SpatioTemporalGraphCoarsener(
        graph=graph,
        alpha=alpha,
        beta=beta,
        P=P,
        radiusCoeff=radiusCoeff,
        depot_id=depot_id
    )
    coarsened_graph, merge_layers = coarsener.coarsen()

    # Solve on the coarsened graph
    coarsened_greedy_solver = GreedySolver(
        graph=coarsened_graph,
        depot_id=depot_id,
        vehicle_capacity=capacity
    )
    coarsened_routes, _ = coarsened_greedy_solver.solve()

    # Inflate the solution back to the original graph
    inflated_routes = coarsener.inflate_routes(coarsened_routes)
    inflated_metrics = calculate_route_metrics(
        graph=graph,
        routes=inflated_routes,
        depot_id=depot_id,
        vehicle_capacity=capacity
    )

    # Return the metrics for this run
    return {
        "alpha": alpha,
        "beta": beta,
        "P": P,
        "radiusCoeff": radiusCoeff,
        "uncoarsened_total_distance": uncoarsened_metrics['total_distance'],
        "uncoarsened_num_vehicles": uncoarsened_metrics['num_vehicles'],
        "uncoarsened_capacity_violations": uncoarsened_metrics['capacity_violations'],
        "uncoarsened_time_violations": uncoarsened_metrics['time_violations'],
        "inflated_total_distance": inflated_metrics['total_distance'],
        "inflated_num_vehicles": inflated_metrics['num_vehicles'],
        "inflated_capacity_violations": inflated_metrics['capacity_violations'],
        "inflated_time_violations": inflated_metrics['time_violations']
    }

def main():
    """
    Main function to run all experiments and save results.
    """
    # Define the ranges of hyperparameters to test
    alpha_values = [0.2, 0.4, 0.6, 0.8, 1.0]
    beta_values = [0.2, 0.4, 0.6, 0.8, 1.0]
    p_values = [0.2, 0.4, 0.6, 0.8, 1.0]
    radiusCoeff_values = [1.0, 1.5, 2.0, 2.5, 3.0]

    csv_file_path = "C101_small.csv" # Example dataset path
    
    results = []
    
    # Iterate through all combinations of hyperparameters
    hyperparameter_combinations = itertools.product(
        alpha_values, beta_values, p_values, radiusCoeff_values
    )
    
    total_runs = len(alpha_values) * len(beta_values) * len(p_values) * len(radiusCoeff_values)
    current_run = 0
    
    for alpha, beta, P, radiusCoeff in hyperparameter_combinations:
        current_run += 1
        logger.info(f"Starting run {current_run}/{total_runs}...")
        experiment_result = run_experiment(
            csv_file_path,
            alpha=alpha,
            beta=beta,
            P=P,
            radiusCoeff=radiusCoeff
        )
        if experiment_result:
            results.append(experiment_result)

    # Save the results to a JSON file
    output_file = "results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)
    logger.info(f"\nAll experiments finished. Results saved to '{output_file}'.")
    logger.info("Now, run 'python analyze_results.py' to generate the box plots.")

if __name__ == "__main__":
    main()
