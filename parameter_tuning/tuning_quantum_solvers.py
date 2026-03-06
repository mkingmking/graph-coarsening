import os
import logging
import random
import json
import time
import sys
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import defaultdict
from pathlib import Path

# --- Path Setup to allow standalone execution ---
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# --- Imports from the Project ---
from graph_coarsening.graph import Graph, compute_euclidean_tau
from graph_coarsening.utils import load_graph_from_csv, calculate_route_metrics
from graph_coarsening.coarsener import SpatioTemporalGraphCoarsener
from graph_coarsening.quantum_solvers.vrp_problem import VRPProblem
from graph_coarsening.quantum_solvers.vrp_solvers import FullQuboSolver, AveragePartitionSolver, IterativeRepairSolver
from graph_coarsening.visualisation import visualize_routes

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# --- Constants & Directories ---
RESULTS_DIR = "tuning_results_quantum_v2"
PLOTS_DIR = os.path.join(RESULTS_DIR, "tuning_plots")
VIS_DIR = os.path.join(RESULTS_DIR, "best_routes_visualizations")

os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(VIS_DIR, exist_ok=True)

# --- Helper Functions (Updated to match New Script Logic) ---

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
    """
    Updated conversion logic from the new script.
    Includes dynamic vehicle sizing to encourage consolidation.
    """
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

    # --- IMPROVEMENT: Optimized Vehicle Count ---
    num_customers = len(customer_ids)
    num_vehicles = max(2, num_customers // 2)  # Half as many vehicles as customers (min 2)
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

# --- Evaluation Logic (Pipeline) ---

def run_evaluation_quantum(
    subgraph: Graph,
    depot_id: str,
    vehicle_capacity: float,
    alpha: float,
    beta: float,
    P: float,
    radiusCoeff: float,
    solver_name: str
) -> tuple[float, dict, list]:
    """
    Runs coarsening -> solving -> inflating.
    Returns: (Objective Score, Metrics Dict, Routes)
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

        # 2. Convert to VRP input
        vrp, int_to_id_map = convert_graph_to_vrp_problem_inputs(coarsened_graph, depot_id, vehicle_capacity)

        # 3. Initialize Solver (Now includes IterativeRepairSolver)
        if solver_name == 'FullQuboSolver':
            solver = FullQuboSolver(vrp)
        elif solver_name == 'AveragePartitionSolver':
            solver = AveragePartitionSolver(vrp)
        elif solver_name == 'IterativeRepairSolver':
            solver = IterativeRepairSolver(vrp)
        else:
            logger.error(f"Invalid solver name: {solver_name}")
            return float('inf'), {}, []

        # 4. Solve using IMPROVED Penalties
        # Tuned parameters - balanced approach from new script
        qubo_params = {
            'only_one': 10_000_000,           # Strong constraint penalty
            'capacity_penalty': 5_000_000,    # Capacity constraints
            'time_window_penalty': 3_000_000, # Time window constraints  
            'vehicle_start_cost': 100_000,    # Encourage fewer vehicles
            'order': 100,                     # Travel distance weight 
            'backend': 'simulated',
            'reads': 5000                     # High quality sampling
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

        # 8. Calculate Objective Score for Tuning
        # Heavy penalties for invalid solutions
        score = metrics['total_distance']
        if not metrics['is_feasible']:
            score += 100_000  # Massive penalty if constraints are broken
        
        # Additional penalties based on metrics
        score += 5000 * metrics['num_vehicles'] # Penalty per vehicle to encourage consolidation
        score += 1000 * metrics['capacity_violations']
        score += 1000 * metrics['time_window_violations']

        return score, metrics, inflated_routes

    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        return float('inf'), {}, []

# --- Plotting Functions ---

def create_boxplots(results_for_plots: list, file_name_only: str, param_name: str, num_customers: int):
    if not results_for_plots: return
    plot_data = pd.DataFrame(results_for_plots)
    if plot_data.empty or 'value' not in plot_data.columns: return

    plt.figure(figsize=(12, 8))
    ax = sns.boxplot(x='value', y='score', data=plot_data, palette='coolwarm')
    ax.set_title(f'Quantum Tuning: {param_name} on {file_name_only} (N={num_customers})')
    ax.set_xlabel(f'{param_name.title()} Value')
    ax.set_ylabel('Objective Score (Lower is Better)')
    plt.tight_layout()
    plot_path = os.path.join(PLOTS_DIR, f"{os.path.splitext(file_name_only)[0]}_{param_name}_boxplot.png")
    plt.savefig(plot_path)
    plt.close()

# --- Main Execution ---

def main():
    parser = argparse.ArgumentParser(description="Tune Quantum Solvers with Graph Coarsening.")
    parser.add_argument("--data", type=str, default=None, help="Directory containing Solomon CSV files.")
    parser.add_argument("--customers", type=int, default=10, help="Number of customers to subsample.")
    parser.add_argument("--trials", type=int, default=20, help="Number of random trials per file.")
    args = parser.parse_args()

    # Determine Dataset Directory
    script_dir = Path(__file__).resolve().parents[1]
    base_dataset_dir = Path(args.data) if args.data else script_dir / "solomon_dataset"

    if not base_dataset_dir.exists():
        logger.error(f"Dataset folder not found: {base_dataset_dir}")
        return
    
    all_flat_results = []
    all_csv_file_paths = sorted(str(p) for p in base_dataset_dir.rglob("*.csv"))

    # Parameter Search Space
    alpha_values = [0.1, 0.3, 0.5, 0.7, 0.9]
    beta_values = [0.1, 0.3, 0.5, 0.7, 0.9]
    P_values = [0.3, 0.5, 0.7]
    radiusCoeff_values = [0.5, 1.0, 1.5, 2.0, 3.0]
    
    # Updated List of Solvers
    quantum_solvers = ['FullQuboSolver', 'AveragePartitionSolver', 'IterativeRepairSolver']

    best_params_per_file = {}

    for csv_file_path in all_csv_file_paths:
        file_name_only = os.path.basename(csv_file_path)
        logger.info(f"\n--- Tuning for {file_name_only} (Subsample: {args.customers}) ---")

        try:
            full_graph, depot_id, VEHICLE_CAPACITY = load_graph_from_csv(csv_file_path)
            initial_graph = create_subgraph(full_graph, depot_id, args.customers)
        except Exception as e:
            logger.error(f"Skipping {csv_file_path}: {e}")
            continue

        best_score_for_file = float('inf')
        best_result_packet = None

        results_for_plots = {
            'alpha': defaultdict(list),
            'beta': defaultdict(list),
            'P': defaultdict(list),
            'radiusCoeff': defaultdict(list),
            'solver_type': defaultdict(list)
        }

        # Random Search Loop
        for i in range(args.trials):
            # Sample parameters
            alpha = random.choice(alpha_values)
            beta = random.choice(beta_values)
            P = random.choice(P_values)
            radiusCoeff = random.choice(radiusCoeff_values)
            solver_type = random.choice(quantum_solvers)

            logger.debug(f"Trial {i+1}/{args.trials}: {solver_type} a={alpha}, b={beta}")

            score, metrics, routes = run_evaluation_quantum(
                initial_graph, depot_id, VEHICLE_CAPACITY,
                alpha, beta, P, radiusCoeff,
                solver_name=solver_type
            )
            
            # Record Data for Plots
            results_for_plots['alpha'][alpha].append(score)
            results_for_plots['beta'][beta].append(score)
            results_for_plots['P'][P].append(score)
            results_for_plots['radiusCoeff'][radiusCoeff].append(score)
            results_for_plots['solver_type'][solver_type].append(score)

            # Record Data for JSON
            row = {
                "file": file_name_only,
                "trial": i,
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

            # Track Best
            if score < best_score_for_file:
                best_score_for_file = score
                best_result_packet = {
                    'params': {'alpha': alpha, 'beta': beta, 'P': P, 'radiusCoeff': radiusCoeff, 'solver': solver_type},
                    'score': score,
                    'metrics': metrics,
                    'routes': routes
                }
                logger.info(f"  New Best ({i}): {solver_type}, Score={score:.2f} (Feasible: {metrics.get('is_feasible')})")

        # --- Post-Processing for this File ---
        
        # 1. Save Intermediate JSON
        with open(os.path.join(RESULTS_DIR, "quantum_tuning_results.json"), "w") as f:
            json.dump(all_flat_results, f, indent=4)

        # 2. Store Summary
        if best_result_packet:
            best_params_per_file[file_name_only] = best_result_packet
            
            # --- DISABLED: Route Visualization to speed up process ---
            # vis_filename = f"BEST_{os.path.splitext(file_name_only)[0]}_{best_result_packet['params']['solver']}.png"
            # vis_path = os.path.join(VIS_DIR, vis_filename)
            # visualize_routes(
            #     initial_graph, 
            #     best_result_packet['routes'], 
            #     depot_id,
            #     title=f"Best found for {file_name_only}\nScore: {best_score_for_file:.2f} | {best_result_packet['params']}",
            #     filename=vis_path
            # )

        # 3. Generate Boxplots (These will still generate)
        for param_name, param_results in results_for_plots.items():
            if param_results:
                plot_data = []
                for value, scores in param_results.items():
                    for s in scores:
                        plot_data.append({'value': value, 'score': s})
                create_boxplots(plot_data, file_name_only, param_name, args.customers)

    # --- Final Summary ---
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

if __name__ == "__main__":
    main()