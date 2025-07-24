import logging
from graph import Graph, compute_euclidean_tau

logger = logging.getLogger(__name__)

class SavingsSolver:
    """
    Implements the Clarke and Wright Savings Algorithm for VRPTW.
    Generates multiple routes.
    """
    def __init__(self, graph: Graph, depot_id: str, vehicle_capacity: float):
        self.graph = graph
        self.depot_id = depot_id
        self.vehicle_capacity = vehicle_capacity

    def _calculate_savings(self) -> list:
        """
        Calculates savings for combining customer pairs (i, j).
        Saving S_ij = C(depot, i) + C(j, depot) - C(i, j)
        where C is travel time (tau).
        
        NOTE: For VRPTW, simple distance savings are often insufficient.
        A more robust savings calculation would consider temporal aspects,
        but for the basic C-W, we use Euclidean distance (travel time).
        """
        savings = []
        customer_ids = [node_id for node_id in self.graph.nodes if node_id != self.depot_id]
        depot_node = self.graph.nodes[self.depot_id]

        for i in range(len(customer_ids)):
            for j in range(i + 1, len(customer_ids)):
                id_i = customer_ids[i]
                id_j = customer_ids[j]
                node_i = self.graph.nodes[id_i]
                node_j = self.graph.nodes[id_j]

                tau_di = compute_euclidean_tau(depot_node, node_i)
                tau_jd = compute_euclidean_tau(node_j, depot_node)
                tau_ij = compute_euclidean_tau(node_i, node_j)
                
                saving = tau_di + tau_jd - tau_ij
                savings.append((saving, id_i, id_j))
        
        # Sort savings in descending order
        savings.sort(key=lambda x: x[0], reverse=True)
        return savings

    def _check_merge_feasibility(self, route1: list, route2: list, merge_point_i: str, merge_point_j: str) -> bool:
        """
        Checks if merging two routes is feasible regarding time windows and capacity.
        Assumes merge_point_i is the end of route1 (before depot) and merge_point_j is start of route2 (after depot).
        
        Args:
            route1 (list): First route (e.g., [D, ..., merge_point_i, D])
            route2 (list): Second route (e.g., [D, merge_point_j, ..., D])
            merge_point_i (str): ID of the last customer in route1 before depot.
            merge_point_j (str): ID of the first customer in route2 after depot.
            
        Returns:
            bool: True if merge is feasible, False otherwise.
        """
        # Form the proposed merged route: [D, ..., route1_nodes_before_i, merge_point_i, merge_point_j, route2_nodes_after_j, ..., D]
        
        # Find index of merge_point_i in route1 (excluding final depot)
        idx_i = -1
        for k in range(len(route1) - 1): # Exclude last depot
            if route1[k] == merge_point_i:
                idx_i = k
                break
        
        # Find index of merge_point_j in route2 (excluding initial depot)
        idx_j = -1
        for k in range(1, len(route2)): # Exclude first depot
            if route2[k] == merge_point_j:
                idx_j = k
                break

        if idx_i == -1 or idx_j == -1: # Should not happen if merge points are valid
            return False

        # Construct the candidate merged route
        # Take route1 up to and including merge_point_i, then route2 from merge_point_j onwards
        candidate_route = route1[:idx_i+1] + route2[idx_j:]
        
        # Check if the combined route starts and ends at depot, if not, fix it
        if candidate_route[0] != self.depot_id:
            candidate_route.insert(0, self.depot_id)
        if candidate_route[-1] != self.depot_id:
            candidate_route.append(self.depot_id)

        # Simulate the route to check feasibility
        current_time = self.graph.nodes[self.depot_id].e
        current_load = 0.0

        for k in range(len(candidate_route) - 1):
            from_node_id = candidate_route[k]
            to_node_id = candidate_route[k+1]

            from_node = self.graph.nodes[from_node_id]
            to_node = self.graph.nodes[to_node_id]

            # Capacity check (only for customer nodes)
            if to_node_id != self.depot_id:
                current_load += to_node.demand
                if current_load > self.vehicle_capacity:
                    return False # Capacity violation

            # Travel time
            travel_time = compute_euclidean_tau(from_node, to_node)
            arrival_time = current_time + travel_time
            
            # Service start time
            service_start_time = max(arrival_time, to_node.e)

            # Time window check (service starts too late)
            if service_start_time > to_node.l:
                return False # Time window violation

            # Update current time
            current_time = service_start_time + to_node.s
        
        # Final check: does the vehicle return to depot within its time window?
        # The last 'current_time' is after serving the last customer.
        # Now, travel back to depot.
        last_customer_node = self.graph.nodes[candidate_route[-2]] # Second to last node is the last customer
        depot_node = self.graph.nodes[self.depot_id]
        travel_time_to_depot = compute_euclidean_tau(last_customer_node, depot_node)
        final_arrival_at_depot = current_time + travel_time_to_depot
        
        # Check if service can start at depot within its time window
        # Assuming depot service time is 0, so arrival is effectively service start
        if final_arrival_at_depot > depot_node.l:
            return False # Depot time window violation

        return True # Merge is feasible

    def solve(self) -> tuple[list, dict]:
        """
        Generates routes using the Clarke and Wright Savings Algorithm.
        
        Returns:
            tuple: A tuple containing:
                - list: A list of generated routes (each route is a list of node IDs).
                - dict: A dictionary of aggregated metrics for all routes.
        """
        logger.info(f"\n--- Starting Savings Solver on graph with depot {self.depot_id} ---")

        # Step 1: Initialize routes (each customer gets its own route)
        routes = {} # {customer_id: [depot_id, customer_id, depot_id]}
        customer_ids = [node_id for node_id in self.graph.nodes if node_id != self.depot_id]
        for cust_id in customer_ids:
            routes[cust_id] = [self.depot_id, cust_id, self.depot_id]
        
        # Map customer ID to the route it belongs to (for quick lookup)
        customer_to_route_map = {cust_id: cust_id for cust_id in customer_ids}

        logger.info(f"  Initial routes: {len(customer_ids)} individual routes.")

        # Step 2: Calculate savings
        savings = self._calculate_savings()
        logger.info(f"  Calculated {len(savings)} potential savings.")

        # Step 3: Iterate through savings and attempt merges
        merged_any_this_iteration = True
        while merged_any_this_iteration:
            merged_any_this_iteration = False
            for saving_value, id_i, id_j in savings:
                # Ensure i and j are still customers and not yet merged into the same route
                if id_i not in customer_to_route_map or id_j not in customer_to_route_map:
                    continue # One or both nodes have already been merged away

                route_id_i = customer_to_route_map[id_i]
                route_id_j = customer_to_route_map[id_j]

                if route_id_i == route_id_j:
                    continue # Already in the same route

                route_i = routes[route_id_i]
                route_j = routes[route_id_j]

                # For Savings algorithm, we merge routes like:
                # Route A: [D, ..., X, i, D]
                # Route B: [D, j, Y, ..., D]
                # Merged: [D, ..., X, i, j, Y, ..., D]
                # So, i must be the last customer in route_i (before depot)
                # and j must be the first customer in route_j (after depot).

                # Check if i is the last customer in route_i (before final depot)
                is_i_endpoint = (route_i[-2] == id_i)
                # Check if j is the first customer in route_j (after initial depot)
                is_j_endpoint = (route_j[1] == id_j)

                # Check if i is the last customer in route_j (before final depot)
                is_i_endpoint_j = (route_j[-2] == id_i)
                # Check if j is the first customer in route_i (after initial depot)
                is_j_endpoint_i = (route_i[1] == id_j)

                # Case 1: Merge Route_i's end with Route_j's start (i -> j)
                if is_i_endpoint and is_j_endpoint:
                    # Check feasibility for merging route_i[:-1] + route_j[1:]
                    if self._check_merge_feasibility(route_i, route_j, id_i, id_j):
                        proposed_merged_route = route_i[:-1] + route_j[1:]
                        new_route_id = f"R_{route_id_i}_{route_id_j}"
                        
                        # Update maps and routes
                        for customer_in_old_route in route_i[1:-1]:
                            customer_to_route_map[customer_in_old_route] = new_route_id
                        for customer_in_old_route in route_j[1:-1]:
                            customer_to_route_map[customer_in_old_route] = new_route_id
                        
                        routes[new_route_id] = proposed_merged_route
                        del routes[route_id_i]
                        del routes[route_id_j]
                        merged_any_this_iteration = True
                        logger.info(f"  Merged routes {route_id_i} and {route_id_j} via ({id_i} -> {id_j}) with saving {saving_value:.2f}. New route: {new_route_id}")
                        break # Restart iteration over savings after a merge
                
                # Case 2: Merge Route_j's end with Route_i's start (j -> i)
                # This is symmetric, so we swap i and j roles
                elif is_j_endpoint_i and is_i_endpoint_j: # This condition is for j at end of route_i and i at start of route_j (reversed logic)
                    # Correct logic for j->i merge:
                    # Route A: [D, ..., X, j, D]
                    # Route B: [D, i, Y, ..., D]
                    # Merged: [D, ..., X, j, i, Y, ..., D]
                    # So, j must be the last customer in route_j (before depot)
                    # and i must be the first customer in route_i (after depot).
                    if route_j[-2] == id_j and route_i[1] == id_i: # Check original roles
                        if self._check_merge_feasibility(route_j, route_i, id_j, id_i): # Pass route_j as first route, route_i as second
                            proposed_merged_route = route_j[:-1] + route_i[1:]
                            new_route_id = f"R_{route_id_j}_{route_id_i}" # Naming convention for the new route
                            
                            # Update maps and routes
                            for customer_in_old_route in route_i[1:-1]:
                                customer_to_route_map[customer_in_old_route] = new_route_id
                            for customer_in_old_route in route_j[1:-1]:
                                customer_to_route_map[customer_in_old_route] = new_route_id
                            
                            routes[new_route_id] = proposed_merged_route
                            del routes[route_id_i]
                            del routes[route_id_j]
                            merged_any_this_iteration = True
                            logger.info(f"  Merged routes {route_id_j} and {route_id_i} via ({id_j} -> {id_i}) with saving {saving_value:.2f}. New route: {new_route_id}")
                            break # Restart iteration over savings after a merge

            # If no merges were made in an entire pass, stop.
            if not merged_any_this_iteration:
                break

        final_routes_list = list(routes.values())
        logger.info(f"--- Savings Solver Finished. Found {len(final_routes_list)} routes. ---")
        
        # The metrics are calculated outside the solver to allow for inflation
        # metrics = calculate_route_metrics(self.graph, final_routes_list, self.depot_id, self.vehicle_capacity)
        return final_routes_list, {} # Return empty dict for metrics, as they're calculated in main

