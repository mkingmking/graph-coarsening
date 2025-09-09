import os
import logging
import random
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from pathlib import Path
import pandas as pd
import json

# Import necessary classes and functions from your project
from graph_coarsening.graph import Graph
from graph_coarsening.utils import load_graph_from_csv, calculate_route_metrics
from graph_coarsening.coarsener import SpatioTemporalGraphCoarsener
from graph_coarsening.greedy_solver import GreedySolver
from graph_coarsening.savings_solver import SavingsSolver


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


def create_boxplots(results_for_plots: list, file_name_only: str, param_name: str):
    """
    Creates boxplots from the tuning results for a single parameter.
    
    Args:
        results_for_plots (list): A list of dictionaries, where each dictionary
                                  represents a single run and contains 'value' and 'score'.
        file_name_only (str): The base name of the file to save the plot.
        param_name (str): The name of the parameter being plotted.
    """
    if not results_for_plots:
        logger.warning(f"No data to plot for {param_name} on {file_name_only}. Skipping boxplot creation.")
        return

    # Convert the list of results to a pandas DataFrame
    plot_data = pd.DataFrame(results_for_plots)

    if plot_data.empty:
        logger.warning(f"DataFrame is empty for {param_name} on {file_name_only}. Skipping boxplot creation.")
        return

    # Ensure the required columns exist
    if 'value' not in plot_data.columns or 'score' not in plot_data.columns:
        logger.error(f"Required columns 'value' or 'score' not found in data for {param_name}.")
        return

    plt.figure(figsize=(12, 8))
    ax = sns.boxplot(x='value', y='score', data=plot_data, palette='coolwarm')
    ax.set_title(f'Performance Scores for {param_name} on {file_name_only}')
    ax.set_xlabel(f'{param_name.title()} Value')
    ax.set_ylabel('Score')
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plot_path = os.path.join("plots", f"{os.path.splitext(file_name_only)[0]}_{param_name}_boxplot.png")
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path)
    logger.info(f"Boxplot saved to {plot_path}")
    plt.close()


def create_scatterplots(results_for_plots: list, file_name_only: str, param_name: str):
    """
    Creates scatterplots from the tuning results for a single parameter.
    
    Args:
        results_for_plots (list): A list of dictionaries, where each dictionary
                                  represents a single run and contains 'value' and 'score'.
        file_name_only (str): The base name of the file to save the plot.
        param_name (str): The name of the parameter being plotted.
    """
    if not results_for_plots:
        logger.warning(f"No data to plot for {param_name} on {file_name_only}. Skipping scatterplot creation.")
        return

    plot_data = pd.DataFrame(results_for_plots)
    
    if plot_data.empty:
        logger.warning(f"DataFrame is empty for {param_name} on {file_name_only}. Skipping scatterplot creation.")
        return

    if 'value' not in plot_data.columns or 'score' not in plot_data.columns:
        logger.error(f"Required columns 'value' or 'score' not found in data for {param_name}.")
        return

    plt.figure(figsize=(12, 8))
    ax = sns.scatterplot(x='value', y='score', data=plot_data, hue='score', palette='viridis', legend=False)
    ax.set_title(f'Performance Scores for {param_name} on {file_name_only}')
    ax.set_xlabel(f'{param_name.title()} Value')
    ax.set_ylabel('Score')
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plot_path = os.path.join("plots", f"{os.path.splitext(file_name_only)[0]}_{param_name}_scatterplot.png")
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path)
    logger.info(f"Scatterplot saved to {plot_path}")
    plt.close()


if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parents[1]  # graph_coarsening/
    base_dataset_dir = ROOT / "solomon_dataset"

    if not base_dataset_dir.exists():
        raise FileNotFoundError(f"Dataset folder not found: {base_dataset_dir}")
    
    # Use rglob for a more efficient way to find all CSV files
    all_csv_file_paths = sorted(str(p) for p in base_dataset_dir.rglob("*.csv"))

    # Define parameter search space for Coarsening
    alpha_values = [0.1, 0.5, 0.9]
    beta_values = [0.1, 0.5, 0.9]
    P_values = [0.3, 0.5, 0.7] # Target percentage of nodes remaining
    radiusCoeff_values = [0.5, 1.0, 1.5, 2.0]
    
    # Solvers to test
    classical_solvers = ['Greedy', 'Savings']

    # Random Search parameters for overall tuning
    num_random_trials_per_file = 1 # Number of random combinations to try

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

        # Dictionary to store all trial results for plotting
        results_for_plots = {
            'alpha': defaultdict(list),
            'beta': defaultdict(list),
            'P': defaultdict(list),
            'radiusCoeff': defaultdict(list),
            'solver_type': defaultdict(list)
        }
        flat_results = []
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
            
            # Store results for boxplot/scatterplot generation
            results_for_plots['alpha'][alpha].append(score)
            results_for_plots['beta'][beta].append(score)
            results_for_plots['P'][P].append(score)
            results_for_plots['radiusCoeff'][radiusCoeff].append(score)
            results_for_plots['solver_type'][solver_type].append(score)

            # --- Add one row into flat_results for JSON ---
            row = {
                "file": file_name_only,
                "alpha": alpha,
                "beta": beta,
                "P": P,
                "radiusCoeff": radiusCoeff,
                "solver_type": solver_type,
                "score": score
            }
            if metrics:
                # Add route metrics with clear prefixes
                row.update({
                    "inflated_total_distance": metrics.get("total_distance"),
                    "num_vehicles": metrics.get("num_vehicles"),
                    "capacity_violations": metrics.get("capacity_violations"),
                    "time_window_violations": metrics.get("time_window_violations"),
                    "is_feasible": metrics.get("is_feasible")
                })
            flat_results.append(row)

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


        # --- Save results_for_plots to JSON (NEW) ---
        json_results = {k: dict(v) for k, v in results_for_plots.items()}
        os.makedirs("results_json", exist_ok=True)
        json_path = os.path.join("results_json", f"{os.path.splitext(file_name_only)[0]}_results.json")
        with open(json_path, "w") as f:
            json.dump(json_results, f, indent=4)
        logger.info(f"Results saved to JSON: {json_path}")
            
        # Generate boxplots for the current file for each parameter
        for param_name, param_results in results_for_plots.items():
            if param_results:
                plot_data = []
                for value, scores in param_results.items():
                    for score in scores:
                        plot_data.append({'value': value, 'score': score})
                create_boxplots(plot_data, file_name_only, param_name)
                # You can also generate scatterplots if desired by uncommenting the line below
                create_scatterplots(plot_data, file_name_only, param_name)


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
