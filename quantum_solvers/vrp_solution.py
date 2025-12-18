class VRPSolution:
    def __init__(self, problem, sample, vehicle_k_limits, solution=None):
        self.problem = problem
        self.depot = self.problem.source_depot
        
        if solution is not None:
            self.solution = solution
        else:
            num_vehicles = len(self.problem.capacities)
            temp_routes = {i: [] for i in range(num_vehicles)}

            for var, val in sample.items():
                if val == 1 and isinstance(var, tuple) and len(var) == 3 and isinstance(var[0], int):
                    i, j, k = var
                    if i < num_vehicles:
                        temp_routes[i].append((k, j))

            final_routes = []
            for i in range(num_vehicles):
                sorted_visits = sorted(temp_routes[i], key=lambda x: x[0])
                route = [j for k, j in sorted_visits]
                if route:
                    final_routes.append(route)
            
           
            self.solution = self._repair_solution(final_routes)
    
    def _calculate_arrival_time(self, route, candidate_node=None):
        """
        Helper to calculate the arrival time at the end of a route, 
        optionally including a candidate node appended to the end.
        Returns float('inf') if any node in the chain is late.
        """
        current_time = 0.0
        
        last_node = self.depot
        
        
        depot_ready = self.problem.time_windows[self.depot][0]
        current_time = max(current_time, depot_ready)
        
        
        full_route = route + ([candidate_node] if candidate_node is not None else [])
        
        for node in full_route:
            
            travel_time = self.problem.costs[last_node][node]
            current_time += travel_time
            
            # Check time window for this node
            ready_time, due_date = self.problem.time_windows[node]
            
            if current_time > due_date:
                return float('inf')  # Violation detected
            
            
            current_time = max(current_time, ready_time)
            
            # Add service time
            current_time += self.problem.service_times[node]
            
            last_node = node
            
        return current_time

    def _repair_solution(self, routes):
        """
        Repairs a solution that may have constraint violations:
        1. Remove duplicate customer visits
        2. Add missing customers
        3. CHECKS TIME WINDOWS before inserting missing customers
        """
        all_customers = set(self.problem.dests)
        visited = set()
        
        # Remove duplicates while preserving first occurrence
        repaired_routes = []
        for route in routes:
            clean_route = []
            for customer in route:
                if customer not in visited:
                    clean_route.append(customer)
                    visited.add(customer)
            if clean_route:  # Only keep non-empty routes
                repaired_routes.append(clean_route)
        
        # Find missing customers
        missing = all_customers - visited
        
        if missing:
            # Try to add missing customers to existing routes
            for customer in missing:
                best_route_idx = -1
                best_cost = float('inf')
                
                # If no routes exist, we must create a new one
                if not repaired_routes:
                    repaired_routes.append([customer])
                    continue
                
                # Find best route to add this customer
                for idx, route in enumerate(repaired_routes):
                    if not route: continue

                    last_customer = route[-1]
                    cost = self.problem.costs[last_customer][customer]
                    
                    # Capacity Constraint ---
                    route_demand = sum(self.problem.weights.get(c, 0) for c in route)
                    customer_demand = self.problem.weights.get(customer, 0)
                    # Handle varying vehicle capacities if they exist, else default to first
                    vehicle_capacity = self.problem.capacities[idx] if idx < len(self.problem.capacities) else self.problem.capacities[0]
                    
                    if route_demand + customer_demand > vehicle_capacity:
                        continue # Skip this vehicle, it's full

                    # Time Window Constraint (NEW) ---
                    # Simulate the route with the new customer added
                    arrival_time = self._calculate_arrival_time(route, candidate_node=customer)
                    if arrival_time == float('inf'):
                        continue # Skip this vehicle, it would be late
                    
                    # If passed both checks, compare cost
                    if cost < best_cost:
                        best_cost = cost
                        best_route_idx = idx
                
                # Add to best route or create new route if no valid vehicle found
                if best_route_idx != -1:
                    repaired_routes[best_route_idx].append(customer)
                else:
                    # Create new route for this customer
                    repaired_routes.append([customer])
        
        return repaired_routes
    
    def check(self):
        capacities = self.problem.capacities
        weights = self.problem.weights
        time_windows = self.problem.time_windows
        service_times = self.problem.service_times
        costs = self.problem.costs

        visited_customers = set()
        for i, route in enumerate(self.solution):
            for customer in route:
                if customer in visited_customers:
                    print(f"Error: Customer {customer} visited more than once.")
                    return False
                visited_customers.add(customer)
            
            current_load = sum(weights.get(dest, 0) for dest in route)
            if i < len(capacities) and current_load > capacities[i]:
                print(f"Error: Vehicle {i} exceeds capacity. Load: {current_load}, Capacity: {capacities[i]}")
                return False

            if not route: continue
            
            current_time = 0.0
            
            # Start from Depot
            # (Assuming start time 0 or window start)
            depot_ready = time_windows[self.depot][0]
            current_time = max(current_time, depot_ready)
            
            # Travel to first node
            current_time += costs[self.depot][route[0]]
            
            ready_time, due_date = time_windows[route[0]]
            if current_time > due_date:
                print(f"Error: Vehicle {i} late for first stop {route[0]}. Arrival: {current_time}, Due: {due_date}")
                return False
            current_time = max(current_time, ready_time)
            current_time += service_times[route[0]]

            for stop_idx in range(len(route) - 1):
                from_node, to_node = route[stop_idx], route[stop_idx+1]
                current_time += costs[from_node][to_node]
                
                ready_time, due_date = time_windows[to_node]
                if current_time > due_date:
                    print(f"Error: Vehicle {i} late for stop {to_node}. Arrival: {current_time:.2f}, Due: {due_date}")
                    return False
                current_time = max(current_time, ready_time)
                current_time += service_times[to_node]

        required_customers = set(self.problem.dests)
        if visited_customers != required_customers:
            missing = required_customers - visited_customers
            print(f"Error: Solution is incomplete. Missing customers: {missing}")
            return False

        return True

    def total_cost(self):
        total_cost = 0
        for route in self.solution:
            if not route: continue
            
            route_cost = self.problem.costs[self.depot][route[0]]
            
            for i in range(len(route) - 1):
                route_cost += self.problem.costs[route[i]][route[i+1]]
            
            route_cost += self.problem.costs[route[-1]][self.depot]
            
            total_cost += route_cost
        return total_cost

    def description(self):
        print("Solution Routes:")
        for i, route in enumerate(self.solution):
            path_str = " -> ".join(map(str, [self.depot] + route + [self.depot]))
            print(f"  Vehicle {i}: {path_str}")
        print(f"\nTotal Cost: {self.total_cost():.2f}")
        print(f"Is Solution Valid (Capacity/TW): {self.check()}")