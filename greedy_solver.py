# greedy_solver.py
import math
from graph import Graph
from node import Node
from utils import compute_euclidean_tau, calculate_route_metrics

class GreedySolver:
    """
    A simple greedy heuristic solver for VRPTW.
    Generates multiple routes if a single vehicle cannot serve all customers.
    """
    def __init__(self, graph: Graph, depot_id: str, vehicle_capacity: float):
        self.graph = graph
        self.depot_id = depot_id
        self.vehicle_capacity = vehicle_capacity

    def solve(self) -> tuple[list, dict]:
        """
        Generates routes on the given graph using a simple greedy heuristic.
        Prioritizes visiting the closest feasible unvisited node.
        Dispatches new vehicles as needed.

        Returns:
            tuple: A tuple containing:
                - list: A list of generated routes (each route is a list of node IDs).
                - dict: A dictionary of aggregated metrics for all routes.
        """
        all_routes = []
        
        all_customers = {node_id for node_id in self.graph.nodes.keys() if node_id != self.depot_id}
        unvisited_customers = set(all_customers)

        print(f"\n--- Starting Greedy Solver on graph with depot {self.depot_id} ---")
        
        vehicle_count = 0
        while unvisited_customers:
            vehicle_count += 1
            current_route = [self.depot_id]
            current_node_id = self.depot_id
            current_time = self.graph.nodes[self.depot_id].e
            current_load = 0.0

            print(f"  Dispatching Vehicle {vehicle_count}. Initial state: Current Node={current_node_id}, Current Time={current_time:.2f}, Load={current_load:.2f}")

            vehicle_made_progress_in_this_route = False

            while True:
                best_next_node_id = None
                min_travel_time = float('inf')
                
                current_node = self.graph.nodes[current_node_id]

                feasible_candidates = []
                for candidate_node_id in unvisited_customers:
                    candidate_node = self.graph.nodes[candidate_node_id]
                    
                    # --- Robust Feasibility Check for Candidate Insertion ---
                    # Temporarily add candidate to route to check full route feasibility
                    temp_route_segment = current_route[1:] + [candidate_node_id]
                    
                    temp_cost, is_feasible_with_candidate = self._get_route_cost_and_feasibility(
                        temp_route_segment, self.vehicle_capacity
                    )

                    if not is_feasible_with_candidate:
                        continue

                    travel_time_to_candidate = compute_euclidean_tau(current_node, candidate_node)
                    
                    if travel_time_to_candidate < min_travel_time:
                        min_travel_time = travel_time_to_candidate
                        best_next_node_id = candidate_node_id
                
                if best_next_node_id:
                    next_node = self.graph.nodes[best_next_node_id]
                    
                    travel_time_to_next = compute_euclidean_tau(current_node, next_node)
                    arrival_time_at_next = current_time + travel_time_to_next
                    service_start_time_at_next = max(arrival_time_at_next, next_node.e)
                    
                    current_time = service_start_time_at_next + next_node.s
                    current_load += next_node.demand
                    
                    current_route.append(best_next_node_id)
                    unvisited_customers.remove(best_next_node_id)
                    current_node_id = best_next_node_id
                    vehicle_made_progress_in_this_route = True
                    print(f"    Vehicle {vehicle_count}: Visited {best_next_node_id}. Current Node={current_node_id}, Current Time={current_time:.2f}, Load={current_load:.2f}")
                else:
                    print(f"    Vehicle {vehicle_count}: No more feasible unvisited customers from {current_node_id}. Ending current route.")
                    break 
            
            if current_node_id != self.depot_id:
                depot_node = self.graph.nodes[self.depot_id]
                current_node = self.graph.nodes[current_node_id]
                travel_time_to_depot = compute_euclidean_tau(current_node, depot_node)
                arrival_time_at_depot = current_time + travel_time_to_depot
                
                if arrival_time_at_depot <= depot_node.l:
                    current_route.append(self.depot_id)
                    print(f"    Vehicle {vehicle_count}: Returned to depot. Route: {current_route}")
                else:
                    print(f"    Vehicle {vehicle_count}: Warning: Cannot return to depot within its time window. Route ends at {current_node_id}. Final time: {arrival_time_at_depot:.2f} (Depot L: {depot_node.l:.2f})")
                    current_route.append(self.depot_id)
            else:
                if current_route[0] != self.depot_id:
                    current_route.insert(0, self.depot_id)
                if current_route[-1] != self.depot_id:
                    current_route.append(self.depot_id)

            if len(current_route) > 2:
                all_routes.append(current_route)
            else:
                print(f"    Vehicle {vehicle_count}: Route was empty or only depot-depot. Not adding to final routes.")
            
            if not unvisited_customers:
                print("  All customers visited.")
                break
            if not vehicle_made_progress_in_this_route and unvisited_customers:
                print(f"  Stuck: Vehicle {vehicle_count} dispatched but could not visit any new customer. {len(unvisited_customers)} customers remaining. Breaking to prevent infinite loop.")
                break
            else:
                print(f"  {len(unvisited_customers)} customers remaining. Dispatching new vehicle.")
        
        print(f"--- Greedy Solver Finished ---")
        
        metrics = calculate_route_metrics(self.graph, all_routes, self.depot_id, self.vehicle_capacity)
        return all_routes, metrics

    def _get_route_cost_and_feasibility(self, route_segment: list, vehicle_capacity: float) -> tuple[float, bool]:
        temp_route_for_metrics = list(route_segment)
        if temp_route_for_metrics and temp_route_for_metrics[0] != self.depot_id:
            temp_route_for_metrics.insert(0, self.depot_id)
        if temp_route_for_metrics and temp_route_for_metrics[-1] != self.depot_id:
            temp_route_for_metrics.append(self.depot_id)
        
        if not temp_route_for_metrics or (len(temp_route_for_metrics) == 2 and temp_route_for_metrics[0] == self.depot_id and temp_route_for_metrics[1] == self.depot_id):
            return 0.0, True

        metrics = calculate_route_metrics(self.graph, [temp_route_for_metrics], self.depot_id, vehicle_capacity)
        return metrics["total_distance"], metrics["is_feasible"]

