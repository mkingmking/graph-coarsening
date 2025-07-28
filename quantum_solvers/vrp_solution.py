class VRPSolution:
    def __init__(self, problem, sample, vehicle_k_limits):
        """
        Initializes a VRPSolution object from a QUBO sample.

        Args:
            problem (VRPProblem): The VRPProblem instance that generated the QUBO.
            sample (dict): A dictionary representing the QUBO solution (e.g., from D-Wave),
                           where keys are (vehicle_idx, node_id, step_k) tuples and values are 0 or 1.
            vehicle_k_limits (list): A list where each element is the maximum step k for that vehicle.
        """
        self.problem = problem # This now contains nodes_dict, depot_id, etc.
        self.vehicle_k_limits = vehicle_k_limits
        
        num_vehicles = len(self.problem.capacities)
        temp_routes = {i: [] for i in range(num_vehicles)}

        # Process the sample from the QUBO solver
        for var, val in sample.items():
            if val == 1:
                i, j_id, k = var # j_id is now the string ID of the node
                temp_routes[i].append((k, j_id))

        final_routes = []
        for i in range(num_vehicles):
            # Sort visits for each vehicle by step (k) to reconstruct the route order
            sorted_visits = sorted(temp_routes[i], key=lambda x: x[0])
            route = [j_id for k, j_id in sorted_visits]
            final_routes.append(route)
        
        self.solution = final_routes

    def check(self):
        """
        Performs a detailed feasibility check on the generated solution,
        including capacity, unique customer visits, and time window adherence.
        This check is for the QUBO-generated solution *before* inflation.
        The ultimate feasibility check is done by calculate_route_metrics on the inflated routes.
        """
        capacities = self.problem.capacities
        customer_demands = self.problem.customer_demands
        nodes_dict = self.problem.nodes_dict
        depot_id = self.problem.depot_id
        time_costs_matrix = self.problem.time_costs_matrix

        all_visited_dests_set = set()
        
        for route_idx, route in enumerate(self.solution):
            current_time = nodes_dict[depot_id].e # Each vehicle starts at depot's earliest time
            current_load = 0.0

            # Simulate the route including depot start and end for feasibility checks
            simulated_route = [depot_id] + route + [depot_id]

            for k in range(len(simulated_route) - 1):
                from_node_id = simulated_route[k]
                to_node_id = simulated_route[k+1]

                from_node = nodes_dict[from_node_id]
                to_node = nodes_dict[to_node_id]

                # Capacity check (only for customer nodes, not depot)
                if to_node_id != depot_id:
                    current_load += customer_demands[to_node_id]
                    if current_load > capacities[route_idx]:
                        print(f"Warning (QUBO Solution Check): Vehicle {route_idx} exceeds capacity. Load: {current_load}, Capacity: {capacities[route_idx]}")
                        return False

                    # Check for duplicate visits across all routes
                    # This check is done when a customer is "visited" (to_node_id is a customer)
                    if to_node_id in all_visited_dests_set:
                        print(f"Warning (QUBO Solution Check): Duplicate customer visit found for customer {to_node_id}.")
                        return False
                    all_visited_dests_set.add(to_node_id)


                # Travel time between current and next node
                travel_time = time_costs_matrix[from_node_id][to_node_id]
                arrival_time = current_time + travel_time
                
                # Service start time at the next node (considering time window)
                service_start_time = max(arrival_time, to_node.e)

                # Time window check (service starts too late)
                if service_start_time > to_node.l:
                    print(f"Warning (QUBO Solution Check): Vehicle {route_idx} violates time window at {to_node_id}. Service starts {service_start_time:.2f} > {to_node.l:.2f}")
                    return False

                # Update current time after service at the current node
                current_time = service_start_time + to_node.s
            
            # Note: The loop iterates up to the last node in simulated_route, which is the depot.
            # So, `current_time` at the end of the loop is the time the vehicle finishes its route
            # and effectively "arrives" at the depot after serving all customers.
            # The depot's time window for return is implicitly checked by the last segment.

        # Check if all required customers were visited by any vehicle
        required_dests_set = set(self.problem.customer_ids)
        if all_visited_dests_set != required_dests_set:
            missing_dests = required_dests_set - all_visited_dests_set
            extra_dests = all_visited_dests_set - required_dests_set
            if missing_dests:
                print(f"Warning (QUBO Solution Check): Missing customer visits: {missing_dests}")
            if extra_dests:
                print(f"Warning (QUBO Solution Check): Extra/unplanned customer visits: {extra_dests}")
            return False

        return True

    def total_cost(self):
        """Calculates the total travel cost for all routes in the solution."""
        costs_matrix = self.problem.costs_matrix
        depot_id = self.problem.depot_id
        total_cost = 0

        for route in self.solution:
            if not route:
                continue # Skip empty routes
            
            # Cost from depot to first customer
            current_route_cost = costs_matrix[depot_id][route[0]]
            
            # Cost between customers
            for i in range(len(route) - 1):
                current_route_cost += costs_matrix[route[i]][route[i+1]]
            
            # Cost from last customer back to depot
            current_route_cost += costs_matrix[route[-1]][depot_id]
            
            total_cost += current_route_cost
            
        return total_cost

    def description(self):
        """Prints a description of the solution, including routes, total cost, and validity."""
        print("Solution Routes (from QUBO, before inflation):")
        for i, route in enumerate(self.solution):
            if route:
                full_path = [self.problem.depot_id] + route + [self.problem.depot_id]
                print(f"  Vehicle {i}: {' -> '.join(map(str, full_path))}")
            else:
                print(f"  Vehicle {i}: Not used")
        print(f"\nTotal Cost (QUBO solution): {self.total_cost():.2f}")
        print(f"Is Solution Valid (Capacity & Visits & TW - QUBO): {self.check()}")

