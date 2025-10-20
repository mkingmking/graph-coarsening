import time
import os
from .graph import Graph
from .utils import load_graph_from_csv
from .quantum_solvers.vrp_problem import VRPProblem
from .quantum_solvers.vrp_solvers import FullQuboSolver, AveragePartitionSolver
from .main2 import convert_graph_to_vrp_problem_inputs, map_solution_to_original_ids

def create_subgraph(original_graph: Graph, depot_id: str, num_customers: int) -> Graph:
    """Creates a new graph with the depot and the first 'num_customers' from the original."""
    
    subgraph = Graph()
    subgraph.add_node(original_graph.nodes[depot_id])
    
    # Get customer nodes, ensuring they are sorted for consistency
    # Sorting by converting to int, as '10' comes before '2' in string sort
    customer_ids = sorted([nid for nid in original_graph.nodes if nid != depot_id], key=int)
    
    # Add the first N customers to the subgraph
    customers_to_include = customer_ids[:num_customers]
    for cid in customers_to_include:
        subgraph.add_node(original_graph.nodes[cid])

    print(f"--- Created a subgraph with 1 depot and {len(customers_to_include)} customers ---")

    # Add edges between all nodes in the new subgraph
    node_ids = list(subgraph.nodes.keys())
    for i in range(len(node_ids)):
        for j in range(i + 1, len(node_ids)):
            id1, id2 = node_ids[i], node_ids[j]
            # Get the edge from the original graph to preserve pre-computed tau
            original_edge = original_graph.get_edge_by_nodes(id1, id2)
            if original_edge:
                subgraph.add_edge(id1, id2, original_edge.tau)
    
    return subgraph

def main():
    """Main function to load a VRP instance from a file and test the quantum solver."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, "C101.csv")
    
    num_customers_to_test = 15  # modify this number

    print(f"--- Loading full graph from {file_path} ---")
    try:
        full_graph, depot_id, capacity = load_graph_from_csv(file_path)
        print(f"Full graph loaded successfully. Depot: {depot_id}, Capacity: {capacity}")
    except Exception as e:
        print(f"Error loading graph: {e}")
        return

    # Create a smaller, solvable subgraph from the loaded file
    test_graph = create_subgraph(full_graph, depot_id, num_customers_to_test)
    
    # Convert the subgraph for the solver
    vrp_problem, int_to_id_map = convert_graph_to_vrp_problem_inputs(test_graph, depot_id, capacity)

    # Use the well-tuned parameters from our previous tests
    qubo_params = {
        'only_one': 2000000,
        'capacity_penalty': 100000,
        'time_window_penalty': 10000,
        'vehicle_start_cost': 5000,
        'order': 1,
        'backend': 'simulated',
        'reads': 2000
    }
    
    print("\n" + "="*50)

    # --- Test FullQuboSolver on the subgraph ---
    print(f"\nRunning FullQuboSolver on a {num_customers_to_test}-customer subset...")
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
    
    # --- Test AveragePartitionSolver on the subgraph ---
    print(f"\nRunning AveragePartitionSolver on a {num_customers_to_test}-customer subset...")
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

