import time
from .graph import Graph, Node, compute_euclidean_tau
from .quantum_solvers.vrp_problem import VRPProblem
from .quantum_solvers.vrp_solvers import FullQuboSolver, AveragePartitionSolver
from .main2 import convert_graph_to_vrp_problem_inputs, map_solution_to_original_ids

def create_toy_graph() -> tuple[Graph, str, float]:
    """Creates a larger, solvable VRP instance for testing."""
    print("--- Creating a test graph (1 depot, 8 customers) ---")
    graph = Graph()
    depot_id = "D0"
    vehicle_capacity = 35.0

    # The depot now has a valid time window and service time
    graph.add_node(Node(id=depot_id, x=0, y=0, s=0, e=0, l=1000, demand=0))
    customers = [
        # Quadrant 1
        Node(id="C1", x=10, y=20, s=5, e=0, l=1000, demand=15),
        Node(id="C2", x=20, y=10, s=5, e=0, l=1000, demand=20),
        # Quadrant 2
        Node(id="C3", x=-10, y=20, s=5, e=0, l=1000, demand=15),
        Node(id="C4", x=-20, y=10, s=5, e=0, l=1000, demand=20),
        # Quadrant 3
        Node(id="C5", x=-10, y=-20, s=5, e=0, l=1000, demand=15),
        Node(id="C6", x=-20, y=-10, s=5, e=0, l=1000, demand=20),
        # Quadrant 4
        Node(id="C7", x=10, y=-20, s=5, e=0, l=1000, demand=15),
        Node(id="C8", x=20, y=-10, s=5, e=0, l=1000, demand=20),
    ]
    for node in customers:
        graph.add_node(node)

    node_ids = list(graph.nodes.keys())
    for i in range(len(node_ids)):
        for j in range(i + 1, len(node_ids)):
            u_node, v_node = graph.nodes[node_ids[i]], graph.nodes[node_ids[j]]
            graph.add_edge(node_ids[i], node_ids[j], compute_euclidean_tau(u_node, v_node))
            
    print(f"Vehicle Capacity: {vehicle_capacity}")
    print("Expected outcome: Approx. 4 routes are needed to respect capacity.")
    return graph, depot_id, vehicle_capacity


def main():
    """Main function to run the quantum solver tests."""
    graph, depot_id, capacity = create_toy_graph()
    vrp_problem, int_to_id_map = convert_graph_to_vrp_problem_inputs(graph, depot_id, capacity)

    # --- Define QUBO parameters ---
    # Final aggressive tuning to force a feasible solution.
    # The 'only_one' penalty is now an order of magnitude larger than all
    # other penalties combined, making it a nearly unbreakable rule.
    # We also dramatically increase reads to explore the larger search space.
    qubo_params = {
        'only_one': 20000000,          # The primary rule, must be dominant.
        'capacity_penalty': 100000,   # Strong secondary rule.
        'time_window_penalty': 10000, # Tertiary rule.
        'vehicle_start_cost': 5000,   # High cost to consolidate routes.
        'order': 1,                   # Baseline for travel distance.
        'backend': 'simulated',
        'reads': 2000                 # Significantly more reads for a harder problem.
    }
    
    print("\n" + "="*50)
    
    # --- Test FullQuboSolver ---
    print("\nRunning FullQuboSolver (with CVRPTW constraints)...")
    start_time_fqs = time.perf_counter()
    fqs = FullQuboSolver(vrp_problem)
    fqs_solution = fqs.solve(
        qubo_params['only_one'], qubo_params['order'], qubo_params['capacity_penalty'],
        qubo_params['time_window_penalty'], qubo_params['vehicle_start_cost'],
        qubo_params['backend'], qubo_params['reads']
    )
    end_time_fqs = time.perf_counter()
    
    print(f"\n--- FullQuboSolver Results ---")
    print(f"Computation Time: {end_time_fqs - start_time_fqs:.4f} seconds")
    fqs_solution.description()
    
    fqs_routes_str = map_solution_to_original_ids(fqs_solution.solution, int_to_id_map)
    print("\nRoutes with Original IDs:")
    for i, route in enumerate(fqs_routes_str):
        print(f"  Vehicle {i}: {depot_id} -> {' -> '.join(route)} -> {depot_id}")
        
    print("\n" + "="*50)
        
    # --- Test AveragePartitionSolver ---
    print("\nRunning AveragePartitionSolver (with CVRPTW constraints)...")
    start_time_aps = time.perf_counter()
    aps = AveragePartitionSolver(vrp_problem)
    aps_solution = aps.solve(
        qubo_params['only_one'], qubo_params['order'], qubo_params['capacity_penalty'],
        qubo_params['time_window_penalty'], qubo_params['vehicle_start_cost'],
        qubo_params['backend'], qubo_params['reads']
    )
    end_time_aps = time.perf_counter()
    
    print(f"\n--- AveragePartitionSolver Results ---")
    print(f"Computation Time: {end_time_aps - start_time_aps:.4f} seconds")
    aps_solution.description()
    
    aps_routes_str = map_solution_to_original_ids(aps_solution.solution, int_to_id_map)
    print("\nRoutes with Original IDs:")
    for i, route in enumerate(aps_routes_str):
        print(f"  Vehicle {i}: {depot_id} -> {' -> '.join(route)} -> {depot_id}")
        
    print("\n\nTest script finished.")


if __name__ == "__main__":
    main()

