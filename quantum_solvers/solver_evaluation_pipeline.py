import logging

# Import classes and functions from separate files
from graph import Graph, compute_euclidean_tau
from utils import calculate_route_metrics

# Import the coarsener
from coarsener import SpatioTemporalGraphCoarsener

# Import the new VRP/QUBO related classes
from vrp_problem_qubo import VRPProblem
from vrp_solution import VRPSolution
from vrp_solvers import FullQuboSolver, AveragePartitionSolver

logger = logging.getLogger(__name__)

def convert_graph_to_vrp_problem_inputs(graph: Graph, depot_id: str, vehicle_capacity: float):
    """
    Converts a Graph object (original or coarsened) into the input format
    expected by the VRPProblem class for QUBO formulation.

    Args:
        graph (Graph): The graph (original or coarsened).
        depot_id (str): The ID of the depot node.
        vehicle_capacity (float): The capacity of each vehicle.

    Returns:
        VRPProblem: An instance of VRPProblem configured with the graph data.
    """
    # Create cost and time_cost matrices as dictionaries of dictionaries
    # This allows using original node IDs (strings) as keys directly.
    costs_matrix = {u_id: {} for u_id in graph.nodes.keys()}
    time_costs_matrix = {u_id: {} for u_id in graph.nodes.keys()}

    for u_id, u_node in graph.nodes.items():
        for v_id, v_node in graph.nodes.items():
            if u_id == v_id:
                costs_matrix[u_id][v_id] = 0.0
                time_costs_matrix[u_id][v_id] = 0.0
            else:
                tau = compute_euclidean_tau(u_node, v_node)
                costs_matrix[u_id][v_id] = tau
                time_costs_matrix[u_id][v_id] = tau # For VRPProblem, time_costs are travel times

    # Prepare VRPProblem specific inputs
    # For the QUBO, we are solving for a fixed number of vehicles.
    # This `num_vehicles_for_qubo` will be the length of the `capacities` list passed to VRPProblem.
    # Let's assume we want to use 3 vehicles for the QUBO part. This can be tuned.
    num_vehicles_for_qubo = 3 # This can be a parameter in your tuning script too!
    capacities_for_qubo = [vehicle_capacity] * num_vehicles_for_qubo

    customer_ids = [node_id for node_id in graph.nodes if node_id != depot_id]
    customer_demands = {node_id: graph.nodes[node_id].demand for node_id in customer_ids}

    # Return the VRPProblem instance
    return VRPProblem(graph.nodes, depot_id, costs_matrix, time_costs_matrix, capacities_for_qubo, customer_ids, customer_demands)


def run_evaluation(initial_graph, depot_id, vehicle_capacity, alpha, beta, P, radiusCoeff, solver_type='Greedy', qubo_solver_type='simulated', num_reads=50, only_one_const=1000, order_const=1, tw_penalty_const=1000):
    """
    Runs the coarsening and solving process for a given set of parameters.
    Returns the objective metric (e.g., total distance) or a penalty if infeasible.
    
    Args:
        initial_graph (Graph): The original graph loaded from the dataset.
        depot_id (str): The ID of the depot node.
        vehicle_capacity (float): The capacity of each vehicle.
        alpha (float): Coarsening parameter for spatial weight.
        beta (float): Coarsening parameter for temporal weight.
        P (float): Coarsening parameter for target graph reduction.
        radiusCoeff (float): Coarsening parameter for merge radius.
        solver_type (str): Type of solver to use ('FullQubo' or 'AveragePartition').
        qubo_solver_type (str): Type of QUBO backend solver ('simulated' or 'dwave').
        num_reads (int): Number of reads for the QUBO solver.
        only_one_const (float): Penalty constant for 'exactly one' constraints in QUBO.
        order_const (float): Weight for travel cost in QUBO objective.
        tw_penalty_const (float): Penalty for time window violations in QUBO.

    Returns:
        tuple: (objective_score, metrics_dict)
               objective_score (float): The calculated objective score (lower is better).
               metrics_dict (dict): A dictionary of detailed route metrics.
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
        coarsened_vrp_problem = convert_graph_to_vrp_problem_inputs(coarsened_graph, depot_id, vehicle_capacity)

        # --- Solve with QUBO Solvers ---
        if solver_type == 'FullQubo':
            qubo_solver = FullQuboSolver(coarsened_vrp_problem)
        elif solver_type == 'AveragePartition':
            qubo_solver = AveragePartitionSolver(coarsened_vrp_problem)
        else:
            raise ValueError("Invalid solver_type. Choose 'FullQubo' or 'AveragePartition'.")

        # Solve the QUBO problem on the coarsened graph
        # Pass the new tw_penalty_const
        coarsened_vrp_solution = qubo_solver.solve(only_one_const, order_const, tw_penalty_const, qubo_solver_type, num_reads)
        
        # The solution from QUBO is based on string IDs of coarsened nodes.
        # The `solution` attribute of VRPSolution already contains string IDs.
        coarsened_routes_from_qubo = coarsened_vrp_solution.solution
        
        # Ensure routes are properly formatted with depot at start/end for inflation
        # The inflation process expects routes to start and end at the depot.
        formatted_coarsened_routes = []
        for route in coarsened_routes_from_qubo:
            if not route: # Skip empty routes
                continue
            temp_route = list(route) # Make a copy
            if temp_route[0] != depot_id:
                temp_route.insert(0, depot_id)
            if temp_route[-1] != depot_id:
                temp_route.append(depot_id)
            
            # Only add if it has more than just the depot (i.e., at least one customer)
            if len(temp_route) > 2:
                formatted_coarsened_routes.append(temp_route)


        # --- Inflate routes back to original graph ---
        final_inflated_routes = coarsener.inflate_route(formatted_coarsened_routes)
        
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

