import os
import logging
import itertools # For grid search
import random # For random search

# Configure logging
# Set to WARNING to reduce console output during tuning trials,
# but INFO will be enabled for the final summary.
logging.basicConfig(level=logging.WARNING, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Import classes and functions from separate files
from graph import Graph, compute_euclidean_tau
from utils import load_graph_from_csv, calculate_route_metrics
from coarsener import SpatioTemporalGraphCoarsener




def convert_graph_to_vrp_problem_inputs(graph: Graph, depot_id: str, vehicle_capacity: float):
    """
    Converts a Graph object (original or coarsened) into the input format
    expected by the VRPProblem class for QUBO formulation.

    Args:
        graph (Graph): The graph (original or coarsened).
        depot_id (str): The ID of the depot node.
        vehicle_capacity (float): The capacity of each vehicle.

    Returns:
        tuple: (sources, costs, time_costs, capacities, dests, weights)
               Note: 'sources' is always [0] as VRPProblem assumes depot is 0.
               Node IDs will be mapped to integers for the cost matrix.
    """
    # Map graph node IDs to integer indices for cost matrix
    # Depot is always 0
    node_id_to_idx = {depot_id: 0}
    current_idx = 1
    customer_nodes_in_graph = [] # Store Node objects for customers

    for node_id, node in graph.nodes.items():
        if node_id != depot_id:
            node_id_to_idx[node_id] = current_idx
            customer_nodes_in_graph.append(node)
            current_idx += 1

    # Sort customer nodes by their original ID for consistent indexing (optional, but good practice)
    # If super-nodes have complex IDs, sorting by actual ID might be less meaningful.
    # For simplicity, we just use the order they were added to customer_nodes_in_graph.

    num_total_nodes = len(graph.nodes) # Includes depot
    
    # Initialize cost matrix (all-to-all Euclidean distances)
    # costs[from_idx][to_idx]
    costs = [[0.0 for _ in range(num_total_nodes)] for _ in range(num_total_nodes)]
    time_costs = [[0.0 for _ in range(num_total_nodes)] for _ in range(num_total_nodes)] # Same as costs for Euclidean

    # Populate cost matrix
    for u_id, u_node in graph.nodes.items():
        for v_id, v_node in graph.nodes.items():
            u_idx = node_id_to_idx[u_id]
            v_idx = node_id_to_idx[v_id]
            if u_id == v_id:
                costs[u_idx][v_idx] = 0.0
                time_costs[u_idx][v_idx] = 0.0
            else:
                tau = compute_euclidean_tau(u_node, v_node)
                costs[u_idx][v_idx] = tau
                time_costs[u_idx][v_idx] = tau # For VRPProblem, time_costs are travel times

    # Prepare VRPProblem specific inputs
    sources = [0] # Depot is always index 0
    capacities = [vehicle_capacity] * 1 # Assuming 1 vehicle for QUBO at this stage, or multiple if you want to model multiple vehicles directly in QUBO
                                        # NOTE: The QUBO formulation itself models multiple vehicles.
                                        # The problem's `capacities` list should reflect the number of vehicles
                                        # that the QUBO is being asked to route.
    
    # For the QUBO, we are solving for a fixed number of vehicles.
    # The `num_vehicles` parameter in VRPProblem will be determined by the length of `capacities`.
    # Let's assume for this integration that the QUBO solvers will try to use `num_vehicles_for_qubo` vehicles.
    # This `num_vehicles_for_qubo` will be the length of the `capacities` list passed to VRPProblem.
    # For now, let's use a fixed number of vehicles for the QUBO part (e.g., 2 or 3, or even more if needed).
    # This is a key parameter that needs to be tuned or decided upon.
    # For demonstration, let's assume we want to use 3 vehicles for the QUBO part.
    num_vehicles_for_qubo = 3 # This can be a parameter in your tuning script too!
    capacities_for_qubo = [vehicle_capacity] * num_vehicles_for_qubo


    # `dests` are the customer indices (not depot)
    dests = [node_id_to_idx[node.id] for node in customer_nodes_in_graph]
    
    # `weights` are the demands of the customers (mapped to their new indices)
    weights = [0.0] * num_total_nodes # Initialize with zeros
    for node_id, node in graph.nodes.items():
        if node_id != depot_id:
            weights[node_id_to_idx[node_id]] = node.demand


    # Return the mapping for inflation purposes: (original_node_id, new_idx)
    idx_to_node_id = {v: k for k, v in node_id_to_idx.items()}

    return VRPProblem(sources, costs, time_costs, capacities_for_qubo, dests, weights), idx_to_node_id


def run_evaluation(initial_graph, depot_id, vehicle_capacity, alpha, beta, P, radiusCoeff, solver_type='Greedy', qubo_solver_type='simulated', num_reads=50, only_one_const=1000, order_const=1):
    """
    Runs the coarsening and solving process for a given set of parameters.
    Returns the objective metric (e.g., total distance) or a penalty if infeasible.
    """
    try:
        # --- Coarsening ---
        coarsener = SpatioTemporalGraphCoarsener(
            graph=initial_graph,
            alpha=alpha,
            beta=beta,
            P=P,
            radiusCoeff=radiusCoeff,
            depot_id=depot_id
        )
        coarsened_graph, merge_layers = coarsener.coarsen()

        # --- Convert Coarsened Graph to VRPProblem for QUBO Solvers ---
        coarsened_vrp_problem, coarsened_idx_to_node_id_map = convert_graph_to_vrp_problem_inputs(coarsened_graph, depot_id, vehicle_capacity)

        # --- Solve with QUBO Solvers ---
        if solver_type == 'FullQubo':
            qubo_solver = FullQuboSolver(coarsened_vrp_problem)
        elif solver_type == 'AveragePartition':
            qubo_solver = AveragePartitionSolver(coarsened_vrp_problem)
        else:
            raise ValueError("Invalid solver_type. Choose 'FullQubo' or 'AveragePartition'.")

        # Solve the QUBO problem on the coarsened graph
        coarsened_vrp_solution = qubo_solver.solve(only_one_const, order_const, qubo_solver_type, num_reads)
        
        # The solution from QUBO is based on integer indices of coarsened nodes.
        # We need to convert these back to string IDs of coarsened nodes.
        coarsened_routes_from_qubo = []
        for route_indices in coarsened_vrp_solution.solution:
            coarsened_route_ids = []
            # Add depot at start if it's not there (VRPProblem assumes 0 is depot)
            if route_indices and route_indices[0] != 0:
                coarsened_route_ids.append(coarsened_idx_to_node_id_map[0]) # Map index 0 back to depot_id
            
            for idx in route_indices:
                coarsened_route_ids.append(coarsened_idx_to_node_id_map[idx])
            
            # Add depot at end if it's not there
            if coarsened_route_ids and coarsened_route_ids[-1] != coarsened_idx_to_node_id_map[0]:
                 coarsened_route_ids.append(coarsened_idx_to_node_id_map[0])
            
            # Ensure routes are not just [depot, depot]
            if len(coarsened_route_ids) > 2:
                coarsened_routes_from_qubo.append(coarsened_route_ids)

        # --- Inflate routes back to original graph ---
        final_inflated_routes = coarsener.inflate_route(coarsened_routes_from_qubo)
        
        # --- Calculate metrics for the inflated route on the ORIGINAL graph ---
        metrics = calculate_route_metrics(initial_graph, final_inflated_routes, depot_id, vehicle_capacity)

        # Define your objective function here.
        # Example: Minimize total distance, heavily penalize violations.
        objective_score = metrics["total_distance"]
        if not metrics["is_feasible"]:
            objective_score += 1000000 # Large penalty for infeasible solutions
        objective_score += metrics["time_window_violations"] * 1000 # Penalty per violation
        objective_score += metrics["capacity_violations"] * 1000 # Penalty per violation
        objective_score += metrics["num_vehicles"] * 100 # Penalty per vehicle (can be adjusted)

        return objective_score, metrics

    except Exception as e:
        logger.error(f"Error during evaluation for params ({alpha}, {beta}, {P}, {radiusCoeff}, solver={solver_type}): {e}")
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

    # Define parameter search space for Coarsening
    alpha_values = [0.1, 0.5, 0.9]
    beta_values = [0.1, 0.5, 0.9]
    P_values = [0.3, 0.5, 0.7] # Target percentage of nodes remaining
    radiusCoeff_values = [0.5, 1.0, 1.5]

    # QUBO Solver specific parameters
    qubo_only_one_const = 1000 # Penalty for 'exactly one' constraints
    qubo_order_const = 1 # Weight for objective function (travel cost)
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
        best_solver_type_for_file = None

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
            # logger.info(f"  Tried: alpha={alpha:.2f}, beta={beta:.2f}, P={P:.2f}, radiusCoeff={radiusCoeff:.2f}, solver={solver_type} Score: {score:.2f}")


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
    logger.info("The current QUBO formulation in VRPProblem does NOT explicitly include time window constraints.")
    logger.info("Solutions from the QUBO solvers are NOT guaranteed to be time-window feasible.")
    logger.info("The 'Time Window Violations' metric in the final results indicates how well the solution")
    logger.info("adheres to time windows when evaluated on the original graph, but the QUBO solver itself")
    logger.info("does not optimize for this constraint.")
    logger.info("To fully solve VRPTW with QUBO, the QUBO formulation needs to be extended with time window penalties.")

