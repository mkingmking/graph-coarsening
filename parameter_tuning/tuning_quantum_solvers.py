import os
import logging
import random # For random search

# Configure logging
# Set to WARNING to reduce console output during tuning trials,
# but INFO will be enabled for the final summary.
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Import classes and functions from separate files
from graph_coarsening.graph import Graph
from graph_coarsening.utils import load_graph_from_csv
from quantum_solvers.solver_evaluation_pipeline import run_evaluation # Import the evaluation pipeline

if __name__ == "__main__":
    base_dataset_dir = 'solomon_dataset'
    all_csv_file_paths = []
    for root, dirs, files in os.walk(base_dataset_dir):
        for file in files:
            if file.endswith('.csv'):
                full_path = os.path.join(root, file)
                all_csv_file_paths.append(full_path)
    all_csv_file_paths.sort()

    # Define parameter search space for Coarsening
    alpha_values = [0,0.1, 0.5, 0.9]
    beta_values = [0.1, 0.5, 0.9]
    P_values = [0.3, 0.5, 0.7] # Target percentage of nodes remaining
    radiusCoeff_values = [0.5, 1.0, 1.5, 2]

    # QUBO Solver specific parameters
    qubo_only_one_const = 1000 # Penalty for 'exactly one' constraints
    qubo_order_const = 1 # Weight for objective function (travel cost)
    qubo_tw_penalty_const = 1000 # New: Penalty for time window violations in QUBO
    qubo_num_reads = 10 # Number of reads for D-Wave solver (or mock)
    qubo_solver_types = ['FullQubo', 'AveragePartition'] # Which QUBO solvers to test

    # Random Search parameters for overall tuning
    num_random_trials_per_file = 20 # Number of random combinations to try for coarsening params

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
        

        # --- Random Search for Coarsening Parameters + QUBO Solver Type ---
        for _ in range(num_random_trials_per_file):
            alpha = random.uniform(0.1, 0.9)
            beta = random.uniform(0.1, 0.9)
            P = random.uniform(0.3, 0.7)
            radiusCoeff = random.uniform(0.5, 2.0)
            solver_type = random.choice(qubo_solver_types) # Randomly choose between FullQubo and AveragePartition

            score, metrics = run_evaluation(
                initial_graph, depot_id, VEHICLE_CAPACITY,
                alpha, beta, P, radiusCoeff,
                solver_type=solver_type, # This is the QUBO solver type
                only_one_const=qubo_only_one_const,
                order_const=qubo_order_const,
                tw_penalty_const=qubo_tw_penalty_const, # Pass the new TW penalty constant
                num_reads=qubo_num_reads
            )

            if score < best_score_for_file:
                best_score_for_file = score
                best_params_for_file = {
                    'alpha': alpha,
                    'beta': beta,
                    'P': P,
                    'radiusCoeff': radiusCoeff,
                    'qubo_solver_type': solver_type # Store which QUBO solver was best
                }
                best_metrics_for_file = metrics
                logger.info(f"  Tried: alpha={alpha:.2f}, beta={beta:.2f}, P={P:.2f}, radiusCoeff={radiusCoeff:.2f}, solver={solver_type} Score: {score:.2f}")


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

    # Ensure INFO level logging is enabled for the final summary
    logger.setLevel(logging.INFO)

    logger.info("\n\n=====================================================================================")
    logger.info("======================== FINAL SUMMARY OF TUNING RESULTS =============================")
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

    logger.info("\nIMPORTANT NOTE ON TIME WINDOWS:")
    logger.info("The QUBO formulation now includes penalty terms for time window violations.")
    logger.info("However, due to the inherent complexity of exact time window modeling in QUBO,")
    logger.info("solutions are still not guaranteed to be perfectly time-window feasible without very high penalties.")
    logger.info("The 'Time Window Violations' metric in the final results indicates how well the solution")
    logger.info("adheres to time windows when evaluated on the original graph.")

