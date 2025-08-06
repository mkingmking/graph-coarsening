# tuning_classical_solvers.py
import os
import logging
import random

# Import necessary classes and functions from your project
from graph_coarsening.graph import Graph
from utils import load_graph_from_csv, calculate_route_metrics
from coarsener import SpatioTemporalGraphCoarsener
from greedy_solver import GreedySolver
from savings_solver import SavingsSolver


# Configure logging for the tuning process
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def run_evaluation_classical(
    initial_graph: Graph,
    depot_id: str,
    vehicle_capacity: float,
    alpha: float,
    beta: float,
    P: float,
    radiusCoeff: float,
    solver_name: str
) -> tuple[float, dict]:
    """
    Runs the full coarsening and solving pipeline for a classical solver
    and returns an objective score and the resulting metrics.
    """
    try:
        # 1. Coarsen the graph with the given parameters
        coarsener = SpatioTemporalGraphCoarsener(
            graph=initial_graph,
            alpha=alpha,
            beta=beta,
            P=P,
            radiusCoeff=radiusCoeff,
            depot_id=depot_id
        )
        coarsened_graph, _ = coarsener.coarsen()

        # 2. Run the specified classical solver on the coarsened graph
        solver = None
        if solver_name == 'Greedy':
            solver = GreedySolver(coarsened_graph, depot_id, vehicle_capacity)
        elif solver_name == 'Savings':
            solver = SavingsSolver(coarsened_graph, depot_id, vehicle_capacity)
        
        if not solver:
            logger.error(f"Invalid solver name provided: {solver_name}")
            return float('inf'), {}

        coarsened_routes, _ = solver.solve()

        # 3. Inflate the solution back to the original graph
        inflated_routes = coarsener.inflate_route(coarsened_routes)

        # 4. Calculate final metrics on the original graph
        metrics = calculate_route_metrics(initial_graph, inflated_routes, depot_id, vehicle_capacity)

        # 5. Calculate a single objective score to optimize
        # This score heavily penalizes violations and the number of vehicles

        #### Hyperparameter selection ####
        score = (
            metrics['total_distance'] +
            1000 * metrics['num_vehicles'] +
            1000 * metrics['capacity_violations'] +
            1000 * metrics['time_window_violations']
        )
        return score, metrics

    except Exception as e:
        logger.error(f"Error during evaluation with params: alpha={alpha:.2f}, beta={beta:.2f}, P={P:.2f}, radiusCoeff={radiusCoeff:.2f}, solver={solver_name}. Error: {e}")
        return float('inf'), {}


if __name__ == "__main__":
    base_dataset_dir = 'solomon_dataset'
    all_csv_file_paths = []
    for root, _, files in os.walk(base_dataset_dir):
        for file in files:
            if file.endswith('.csv'):
                full_path = os.path.join(root, file)
                all_csv_file_paths.append(full_path)
    all_csv_file_paths.sort()

    # Define parameter search space for Coarsening
    alpha_values = [0.1, 0.5, 0.9]
    beta_values = [0.1, 0.5, 0.9]
    P_values = [0.3, 0.5, 0.7] # Target percentage of nodes remaining
    radiusCoeff_values = [0.5, 1.0, 1.5, 2.0]
    
    # Solvers to test
    classical_solvers = ['Greedy', 'Savings']

    # Random Search parameters for overall tuning
    num_random_trials_per_file = 20 # Number of random combinations to try

    best_params_per_file = {}

    for csv_file_path in all_csv_file_paths:
        file_name_only = os.path.basename(csv_file_path)
        logger.info(f"\n--- Tuning parameters for {file_name_only} ---")

        try:
            initial_graph, depot_id, VEHICLE_CAPACITY = load_graph_from_csv(csv_file_path)
        except Exception as e:
            logger.error(f"Skipping {csv_file_path} due to error loading graph: {e}")
            continue

        best_score_for_file = float('inf')
        best_params_for_file = None
        best_metrics_for_file = None
        
        # --- Random Search for Coarsening Parameters + Solver Type ---
        for _ in range(num_random_trials_per_file):
            alpha = random.choice(alpha_values)
            beta = random.choice(beta_values)
            P = random.choice(P_values)
            radiusCoeff = random.choice(radiusCoeff_values)
            solver_type = random.choice(classical_solvers)

            score, metrics = run_evaluation_classical(
                initial_graph, depot_id, VEHICLE_CAPACITY,
                alpha, beta, P, radiusCoeff,
                solver_name=solver_type
            )

            if score < best_score_for_file:
                best_score_for_file = score
                best_params_for_file = {
                    'alpha': alpha,
                    'beta': beta,
                    'P': P,
                    'radiusCoeff': radiusCoeff,
                    'solver_type': solver_type
                }
                best_metrics_for_file = metrics
                logger.info(f"  Best so far: alpha={alpha:.2f}, beta={beta:.2f}, P={P:.2f}, radiusCoeff={radiusCoeff:.2f}, solver={solver_type} Score: {score:.2f}")


        if best_params_for_file:
            best_params_per_file[file_name_only] = {
                'params': best_params_for_file,
                'score': best_score_for_file,
                'metrics': best_metrics_for_file
            }
            logger.info(f"Best params for {file_name_only}: {best_params_for_file} with score {best_score_for_file:.2f}")
            if best_metrics_for_file:
                logger.info(f"  Total Distance: {best_metrics_for_file['total_distance']:.2f}, Num Vehicles: {best_metrics_for_file['num_vehicles']}, Feasible: {best_metrics_for_file['is_feasible']}")
        else:
            logger.warning(f"No feasible parameters found for {file_name_only}")


    logger.info("\n\n=====================================================================================")
    logger.info("======================== FINAL SUMMARY OF CLASSICAL TUNING RESULTS =============================")
    logger.info("=====================================================================================")

    for file_name, result in best_params_per_file.items():
        logger.info(f"File: {file_name}")
        logger.info(f"  Best Coarsening Params: {result['params']}")
        logger.info(f"  Best Objective Score: {result['score']:.2f}")
        if result['metrics']:
            logger.info(f"  Resulting Metrics (on Original Graph after Inflation):")
            for key, value in result['metrics'].items():
                if isinstance(value, float):
                    logger.info(f"    {key.replace('_', ' ').title()}: {value:.2f}")
                else:
                    logger.info(f"    {key.replace('_', ' ').title()}: {value}")
        logger.info("-" * 80)
