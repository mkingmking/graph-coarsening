# In your main.py or a new tuning_script.py

import os
import logging
import itertools # For grid search
import random # For random search

# Configure logging
logging.basicConfig(level=logging.WARNING, format='%(levelname)s: %(message)s') # Set to WARNING to reduce console output during tuning
logger = logging.getLogger(__name__)

# Import your classes and functions
from graph import Graph
from utils import load_graph_from_csv, calculate_route_metrics
from coarsener import SpatioTemporalGraphCoarsener
from greedy_solver import GreedySolver
from savings_solver import SavingsSolver

def run_evaluation(initial_graph, depot_id, vehicle_capacity, alpha, beta, P, radiusCoeff, solver_type='Greedy'):
    """
    Runs the coarsening and solving process for a given set of parameters.
    Returns the objective metric (e.g., total distance) or a penalty if infeasible.
    """
    try:
        coarsener = SpatioTemporalGraphCoarsener(
            graph=initial_graph,
            alpha=alpha,
            beta=beta,
            P=P,
            radiusCoeff=radiusCoeff,
            depot_id=depot_id
        )
        coarsened_graph, merge_layers = coarsener.coarsen()

        if solver_type == 'Greedy':
            solver = GreedySolver(coarsened_graph, depot_id, vehicle_capacity)
        elif solver_type == 'Savings':
            solver = SavingsSolver(coarsened_graph, depot_id, vehicle_capacity)
        else:
            raise ValueError("Invalid solver_type. Choose 'Greedy' or 'Savings'.")

        coarsened_routes, _ = solver.solve()
        final_inflated_routes = coarsener.inflate_route(coarsened_routes)
        
        metrics = calculate_route_metrics(initial_graph, final_inflated_routes, depot_id, vehicle_capacity)

        # Define your objective function here.
        # Example: Minimize total distance, heavily penalize violations.
        objective_score = metrics["total_distance"]
        if not metrics["is_feasible"]:
            objective_score += 1000000 # Large penalty for infeasible solutions
        objective_score += metrics["time_window_violations"] * 1000 # Penalty per violation
        objective_score += metrics["capacity_violations"] * 1000 # Penalty per violation
        objective_score += metrics["num_vehicles"] * 100 # Penalty per vehicle

        return objective_score, metrics

    except Exception as e:
        logger.error(f"Error during evaluation for params ({alpha}, {beta}, {P}, {radiusCoeff}): {e}")
        return float('inf'), {} # Return a very high score for errors/failures

if __name__ == "__main__":
    base_dataset_dir = 'solomon_dataset'
    all_csv_file_paths = []
    for root, dirs, files in os.walk(base_dataset_dir):
        for file in files:
            if file.endswith('.csv'):
                full_path = os.path.join(root, file)
                all_csv_file_paths.append(full_path)
    all_csv_file_paths.sort()

    # Define parameter search space
    # For Grid Search:
    alpha_values = [0.1, 0.3, 0.5, 0.7, 0.9]
    beta_values = [0.1, 0.3, 0.5, 0.7, 0.9]
    P_values = [0.3, 0.5, 0.7] # Target percentage of nodes remaining
    radiusCoeff_values = [0.5, 1.0, 1.5, 2.0]

    # For Random Search (define ranges and number of trials):
    num_random_trials = 50 # Number of random combinations to try
    
    best_params_per_file = {}

    for csv_file_path in all_csv_file_paths:
        file_name_only = os.path.basename(csv_file_path)
        logger.info(f"\n--- Tuning parameters for {file_name_only} ---")

        try:
            initial_graph, depot_id, VEHICLE_CAPACITY = load_graph_from_csv(csv_file_path)
        except Exception as e:
            logger.error(f"Skipping {csv_file_path} due to error loading graph: {e}")
            continue

        best_score = float('inf')
        best_params = None
        best_metrics = None

        # --- Grid Search Example ---
        # for alpha, beta, P, radiusCoeff in itertools.product(alpha_values, beta_values, P_values, radiusCoeff_values):
        #     score, metrics = run_evaluation(initial_graph, depot_id, VEHICLE_CAPACITY, alpha, beta, P, radiusCoeff, solver_type='Greedy')
        #     if score < best_score:
        #         best_score = score
        #         best_params = {'alpha': alpha, 'beta': beta, 'P': P, 'radiusCoeff': radiusCoeff}
        #         best_metrics = metrics
        #     logger.info(f"  Tried: {best_params} Score: {score:.2f}")

        # --- Random Search Example ---
        for _ in range(num_random_trials):
            alpha = random.uniform(0.1, 0.9)
            beta = random.uniform(0.1, 0.9)
            P = random.uniform(0.3, 0.7)
            radiusCoeff = random.uniform(0.5, 2.0)

            score, metrics = run_evaluation(initial_graph, depot_id, VEHICLE_CAPACITY, alpha, beta, P, radiusCoeff, solver_type='Greedy')
            if score < best_score:
                best_score = score
                best_params = {'alpha': alpha, 'beta': beta, 'P': P, 'radiusCoeff': radiusCoeff}
                best_metrics = metrics
            # logger.info(f"  Tried: alpha={alpha:.2f}, beta={beta:.2f}, P={P:.2f}, radiusCoeff={radiusCoeff:.2f} Score: {score:.2f}")


        if best_params:
            best_params_per_file[file_name_only] = {
                'params': best_params,
                'score': best_score,
                'metrics': best_metrics
            }
            logger.info(f"Best params for {file_name_only}: {best_params} with score {best_score:.2f}")
            if best_metrics:
                logger.info(f"  Total Distance: {best_metrics['total_distance']:.2f}, Num Vehicles: {best_metrics['num_vehicles']}, Feasible: {best_metrics['is_feasible']}")
        else:
            logger.warning(f"No feasible parameters found for {file_name_only}")

    logger.info("\n\n--- Summary of Best Parameters per File ---")
    for file_name, result in best_params_per_file.items():
        logger.info(f"File: {file_name}")
        logger.info(f"  Best Parameters: {result['params']}")
        logger.info(f"  Best Score (Objective): {result['score']:.2f}")
        if result['metrics']:
            logger.info(f"  Resulting Metrics: Total Distance={result['metrics']['total_distance']:.2f}, Num Vehicles={result['metrics']['num_vehicles']}, Feasible={result['metrics']['is_feasible']}")
        logger.info("-" * 30)