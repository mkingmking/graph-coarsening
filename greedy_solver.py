import logging
from graph import Graph, compute_euclidean_tau
from utils import calculate_route_metrics # Import calculate_route_metrics

logger = logging.getLogger(__name__)

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
        
        # All nodes in the graph that are not the depot
        all_customers = {node_id for node_id in self.graph.nodes.keys() if node_id != self.depot_id}
        unvisited_customers = set(all_customers) # Customers remaining to be visited

        logger.info(f"\n--- Starting Greedy Solver on graph with depot {self.depot_id} ---")
        
        vehicle_count = 0
        while unvisited_customers:
            vehicle_count += 1
            current_route = [self.depot_id]
            current_node_id = self.depot_id
            current_time = self.graph.nodes[self.depot_id].e # Each vehicle starts at depot's earliest time window
            current_load = 0.0 # Current load of the vehicle

            logger.info(f"  Dispatching Vehicle {vehicle_count}. Initial state: Current Node={current_node_id}, Current Time={current_time:.2f}, Load={current_load:.2f}")

            vehicle_made_progress_in_this_route = False # Flag to track if the current vehicle adds any customers

            while True:
                best_next_node_id = None
                min_travel_time = float('inf')
                
                current_node = self.graph.nodes[current_node_id]

                # Find the closest feasible unvisited customer for the current vehicle
                feasible_candidates = []
                for candidate_node_id in unvisited_customers:
                    candidate_node = self.graph.nodes[candidate_node_id]

                    # Check capacity
                    if current_load + candidate_node.demand > self.vehicle_capacity:
                        continue  # Cannot add this customer due to capacity

                    travel_time_to_candidate = compute_euclidean_tau(current_node, candidate_node)
                    arrival_time_at_candidate = current_time + travel_time_to_candidate
                    service_start_time_at_candidate = max(arrival_time_at_candidate, candidate_node.e)

                    # Check time window for the candidate node itself
                    if service_start_time_at_candidate > candidate_node.l:
                        continue
                    
                    # Re-introducing: Check if returning to depot from this candidate is feasible
                    # This is a critical check for a greedy approach to ensure overall route feasibility
                    # when deciding the *next* customer.
                    finish_time_at_candidate = service_start_time_at_candidate + candidate_node.s
                    depot_node = self.graph.nodes[self.depot_id]
                    travel_back_to_depot = compute_euclidean_tau(candidate_node, depot_node)
                    arrival_back_to_depot = finish_time_at_candidate + travel_back_to_depot

                    if arrival_back_to_depot <= depot_node.l:
                        feasible_candidates.append((travel_time_to_candidate, candidate_node_id))
                
                if feasible_candidates:
                    # Sort feasible candidates by travel time to find the closest
                    feasible_candidates.sort(key=lambda x: x[0])
                    min_travel_time, best_next_node_id = feasible_candidates[0]

                if best_next_node_id:
                    next_node = self.graph.nodes[best_next_node_id]
                    
                    travel_time_to_next = compute_euclidean_tau(current_node, next_node)
                    arrival_time_at_next = current_time + travel_time_to_next
                    service_start_time_at_next = max(arrival_time_at_next, next_node.e)
                    
                    current_time = service_start_time_at_next + next_node.s
                    current_load += next_node.demand # Update load
                    
                    current_route.append(best_next_node_id)
                    unvisited_customers.remove(best_next_node_id)
                    current_node_id = best_next_node_id
                    vehicle_made_progress_in_this_route = True # Customer added!
                    logger.info(f"    Vehicle {vehicle_count}: Visited {best_next_node_id}. Current Node={current_node_id}, Current Time={current_time:.2f}, Load={current_load:.2f}")
                else:
                    logger.info(f"    Vehicle {vehicle_count}: No more feasible unvisited customers from {current_node_id}. Ending current route.")
                    break 
            
            # Current vehicle returns to depot
            if current_node_id != self.depot_id:
                depot_node = self.graph.nodes[self.depot_id]
                current_node = self.graph.nodes[current_node_id]
                travel_time_to_depot = compute_euclidean_tau(current_node, depot_node)
                arrival_time_at_depot = current_time + travel_time_to_depot
                
                if arrival_time_at_depot > depot_node.l:
                    logger.warning(
                        f"    Vehicle {vehicle_count}: WARNING: Final route segment cannot return to depot within its time window. Route: {current_route}. Arrival at depot: {arrival_time_at_depot:.2f} (Depot L: {depot_node.l:.2f})"
                    )
                current_route.append(self.depot_id)
                logger.info(f"    Vehicle {vehicle_count}: Returned to depot. Route: {current_route}")
            else:
                if current_route[0] != self.depot_id:
                    current_route.insert(0, self.depot_id)
                if current_route[-1] != self.depot_id:
                    current_route.append(self.depot_id)

            # Only add route if it actually contains customers (not just [depot, depot])
            if len(current_route) > 2: # Route must have at least one customer
                all_routes.append(current_route)
            else:
                logger.info(f"    Vehicle {vehicle_count}: Route contains no customers, skipping: {current_route}")
            
            # Check for infinite loop condition: if a vehicle was dispatched but visited no customers,
            # and there are still unvisited customers, it means we are stuck.
            if not vehicle_made_progress_in_this_route and unvisited_customers:
                logger.warning(f"  Stuck: Vehicle {vehicle_count} dispatched but could not visit any customer. {len(unvisited_customers)} customers remaining. Breaking to prevent infinite loop.")
                break # Break the outer loop to prevent infinite vehicle dispatch

            if not unvisited_customers:
                logger.info("  All customers visited.")
                break
            else:
                logger.info(f"  {len(unvisited_customers)} customers remaining. Dispatching new vehicle.")
        
        logger.info(f"--- Greedy Solver Finished ---")
        
        metrics = calculate_route_metrics(self.graph, all_routes, self.depot_id, self.vehicle_capacity)
        return all_routes, metrics

