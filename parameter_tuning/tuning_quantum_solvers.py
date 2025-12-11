import os
import logging
import random
import json
import time
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import defaultdict
from pathlib import Path
import sys

current_file = Path(__file__).resolve()

project_root = current_file.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import necessary classes from your project
from graph_coarsening.graph import Graph, compute_euclidean_tau
from graph_coarsening.utils import load_graph_from_csv, calculate_route_metrics
from graph_coarsening.coarsener import SpatioTemporalGraphCoarsener
from graph_coarsening.quantum_solvers.vrp_problem import VRPProblem
from graph_coarsening.quantum_solvers.vrp_solvers import FullQuboSolver, AveragePartitionSolver

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Constants
NUM_CUSTOMERS_SUBGRAPH = 10  # The limit we established for feasible quantum solving
RESULTS_DIR = "tuning_results_quantum"
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# --- Helper Functions from main_quantum.py ---

def create_subgraph(original_graph: Graph, depot_id: str, num_customers: int) -> Graph:
    """Creates a new graph with the depot and the first 'num_customers' from the original."""
    subgraph = Graph()
    subgraph.add_node(original_graph.nodes[depot_id])
    
    # Sort by int to ensure deterministic selection (1, 2, 3...)
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

def convert_graph_to_vrp_problem_inputs(graph: Graph, depot_id: str, vehicle_capacity: float) -> tuple[VRPProblem, list]:
    customer_ids = sorted([nid for nid in graph.nodes if nid != depot_id]) # String sort for supernodes
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
    mapped_routes = []
    for route_int in solution_routes_int:
        if route_int:
            mapped_routes.append([int_to_id_map[i] for i in route_int])
    return mapped_routes

# --- Evaluation Logic ---

def run_evaluation_quantum(
    subgraph: Graph,
    depot_id: str,
    vehicle_capacity: float,
    alpha: float,
    beta: float,
    P: float,
    radiusCoeff: float,
    solver_name: str
) -> tuple[float, dict]:
    """
    Runs the full coarsening and solving pipeline using QUANTUM solvers.
    """
    try:
        # 1. Coarsen the graph
        coarsener = SpatioTemporalGraphCoarsener(
            graph=subgraph,
            alpha=alpha,
            beta=beta,
            P=P,
            radiusCoeff=radiusCoeff,
            depot_id=depot_id
        )
        coarsened_graph, _ = coarsener.coarsen()

        # 2. Convert to VRP input for Quantum Solver
        vrp, int_to_id_map = convert_graph_to_vrp_problem_inputs(coarsened_graph, depot_id, vehicle_capacity)

        # 3. Initialize Solver
        if solver_name == 'FullQuboSolver':
            solver = FullQuboSolver(vrp)
        elif solver_name == 'AveragePartitionSolver':
            solver = AveragePartitionSolver(vrp)
        else:
            logger.error(f"Invalid solver name: {solver_name}")
            return float('inf'), {}

        # 4. Solve using tuned penalties
        # These constants are critical for feasibility on larger graphs
        qubo_params = {
            'only_one': 2000000, 
            'capacity_penalty': 100000,
            'time_window_penalty': 10000, 
            'vehicle_start_cost': 5000,
            'order': 1, 
            'backend': 'simulated', 
            'reads': 2000
        }

        sol = solver.solve(
            qubo_params['only_one'], qubo_params['order'], qubo_params['capacity_penalty'],
            qubo_params['time_window_penalty'], qubo_params['vehicle_start_cost'],
            qubo_params['backend'], qubo_params['reads']
        )

        # 5. Parse and Format Routes
        solution_routes_str = map_solution_to_original_ids(sol.solution, int_to_id_map)
        
        formatted_coarsened_routes = []
        for r in solution_routes_str:
            if r:
                tmp = [depot_id] + r + [depot_id]
                if len(tmp) > 2:
                    formatted_coarsened_routes.append(tmp)

        # 6. Inflate back to original subgraph
        inflated_routes = coarsener.inflate_route(formatted_coarsened_routes)

        # 7. Calculate metrics on the original subgraph
        metrics = calculate_route_metrics(subgraph, inflated_routes, depot_id, vehicle_capacity)

        # 8. Calculate Objective Score
        # Heavy penalties for invalid solutions
        score = metrics['total_distance']
        if not metrics['is_feasible']:
            score += 100000  # Massive penalty if constraints are broken
        
        score += 1000 * metrics['num_vehicles']
        score += 1000 * metrics['capacity_violations']
        score += 1000 * metrics['time_window_violations']

        return score, metrics

    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        return float('inf'), {}

# --- Plotting Functions ---

def create_boxplots(results_for_plots: list, file_name_only: str, param_name: str):
    if not results_for_plots: return
    plot_data = pd.DataFrame(results_for_plots)
    if plot_data.empty or 'value' not in plot_data.columns: return

    plt.figure(figsize=(12, 8))
    ax = sns.boxplot(x='value', y='score', data=plot_data, palette='coolwarm')
    ax.set_title(f'Quantum Tuning: {param_name} on {file_name_only} (Subsample {NUM_CUSTOMERS_SUBGRAPH})')
    ax.set_xlabel(f'{param_name.title()} Value')
    ax.set_ylabel('Objective Score')
    plt.tight_layout()
    plot_path = os.path.join(PLOTS_DIR, f"{os.path.splitext(file_name_only)[0]}_{param_name}_boxplot.png")
    plt.savefig(plot_path)
    plt.close()

def create_scatterplots(results_for_plots: list, file_name_only: str, param_name: str):
    if not results_for_plots: return
    plot_data = pd.DataFrame(results_for_plots)
    if plot_data.empty or 'value' not in plot_data.columns: return

    plt.figure(figsize=(12, 8))
    ax = sns.scatterplot(x='value', y='score', data=plot_data, hue='score', palette='viridis', legend=False)
    ax.set_title(f'Quantum Tuning: {param_name} on {file_name_only} (Subsample {NUM_CUSTOMERS_SUBGRAPH})')
    ax.set_xlabel(f'{param_name.title()} Value')
    ax.set_ylabel('Objective Score')
    plt.tight_layout()
    plot_path = os.path.join(PLOTS_DIR, f"{os.path.splitext(file_name_only)[0]}_{param_name}_scatterplot.png")
    plt.savefig(plot_path)
    plt.close()

# --- Main Execution ---

if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parents[1]
    base_dataset_dir = script_dir / "solomon_dataset"

    if not base_dataset_dir.exists():
        raise FileNotFoundError(f"Dataset folder not found: {base_dataset_dir}")
    
    all_flat_results = []
    all_csv_file_paths = sorted(str(p) for p in base_dataset_dir.rglob("*.csv"))

    # Parameter Search Space
    alpha_values = [0.1, 0.5, 0.9]
    beta_values = [0.1, 0.5, 0.9]
    P_values = [0.3, 0.5, 0.7]
    radiusCoeff_values = [0.5, 1.0, 1.5, 2.0]
    
    # Quantum Solvers to test
    quantum_solvers = ['FullQuboSolver', 'AveragePartitionSolver']

    num_random_trials_per_file = 20
    best_params_per_file = {}

    for csv_file_path in all_csv_file_paths:
        file_name_only = os.path.basename(csv_file_path)
        logger.info(f"\n--- Tuning Quantum Solvers for {file_name_only} (First {NUM_CUSTOMERS_SUBGRAPH} customers) ---")

        try:
            full_graph, depot_id, VEHICLE_CAPACITY = load_graph_from_csv(csv_file_path)
            # Create the computationally feasible subgraph
            initial_graph = create_subgraph(full_graph, depot_id, NUM_CUSTOMERS_SUBGRAPH)
        except Exception as e:
            logger.error(f"Skipping {csv_file_path}: {e}")
            continue

        best_score_for_file = float('inf')
        best_params_for_file = None
        best_metrics_for_file = None

        # Data collection for plots
        results_for_plots = {
            'alpha': defaultdict(list),
            'beta': defaultdict(list),
            'P': defaultdict(list),
            'radiusCoeff': defaultdict(list),
            'solver_type': defaultdict(list)
        }

        for _ in range(num_random_trials_per_file):
            alpha = random.choice(alpha_values)
            beta = random.choice(beta_values)
            P = random.choice(P_values)
            radiusCoeff = random.choice(radiusCoeff_values)
            solver_type = random.choice(quantum_solvers)

            score, metrics = run_evaluation_quantum(
                initial_graph, depot_id, VEHICLE_CAPACITY,
                alpha, beta, P, radiusCoeff,
                solver_name=solver_type
            )
            
            # Store for plotting
            results_for_plots['alpha'][alpha].append(score)
            results_for_plots['beta'][beta].append(score)
            results_for_plots['P'][P].append(score)
            results_for_plots['radiusCoeff'][radiusCoeff].append(score)
            results_for_plots['solver_type'][solver_type].append(score)

            # Store for JSON
            row = {
                "file": file_name_only,
                "alpha": alpha, "beta": beta, "P": P, "radiusCoeff": radiusCoeff,
                "solver_type": solver_type, "score": score
            }
            if metrics:
                row.update({
                    "inflated_total_distance": metrics.get("total_distance"),
                    "num_vehicles": metrics.get("num_vehicles"),
                    "is_feasible": metrics.get("is_feasible"),
                    "route_duration": metrics.get("total_route_duration")
                })
            all_flat_results.append(row)

            if score < best_score_for_file:
                best_score_for_file = score
                best_params_for_file = {
                    'alpha': alpha, 'beta': beta, 'P': P, 'radiusCoeff': radiusCoeff, 'solver': solver_type
                }
                best_metrics_for_file = metrics
                logger.info(f"  New Best: params={best_params_for_file}, Score={score:.2f}")

        # Save interim results
        combined_json_path = os.path.join(RESULTS_DIR, "quantum_tuning_results.json")
        with open(combined_json_path, "w") as f:
            json.dump(all_flat_results, f, indent=4)

        best_params_per_file[file_name_only] = {
            'params': best_params_for_file,
            'score': best_score_for_file,
            'metrics': best_metrics_for_file
        }

        # Generate plots
        for param_name, param_results in results_for_plots.items():
            if param_results:
                plot_data = []
                for value, scores in param_results.items():
                    for s in scores:
                        plot_data.append({'value': value, 'score': s})
                create_boxplots(plot_data, file_name_only, param_name)

    # Final Summary Log
    logger.info("\n" + "="*60)
    logger.info("FINAL SUMMARY OF QUANTUM TUNING RESULTS")
    logger.info("="*60)
    for file_name, result in best_params_per_file.items():
        logger.info(f"File: {file_name}")
        logger.info(f"  Best Params: {result['params']}")
        logger.info(f"  Best Score: {result['score']:.2f}")
        if result['metrics']:
            m = result['metrics']
            logger.info(f"  Metrics: Feasible={m.get('is_feasible')}, Dist={m.get('total_distance'):.2f}, Vehicles={m.get('num_vehicles')}")
        logger.info("-" * 40)