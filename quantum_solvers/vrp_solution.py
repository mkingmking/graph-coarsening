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
                # FIX: This now checks that the variable is a valid routing variable
                # (a tuple of 3 integers) and explicitly ignores other variable types,
                # such as the slack variables ('s', i, m) from the capacity constraint.
                if val == 1 and isinstance(var, tuple) and len(var) == 3 and isinstance(var[0], int):
                    i, j, k = var
                    # Safety check to ensure the vehicle index is valid
                    if i < num_vehicles:
                        temp_routes[i].append((k, j))

            final_routes = []
            for i in range(num_vehicles):
                # Sort visits by step 'k' to form the route
                sorted_visits = sorted(temp_routes[i], key=lambda x: x[0])
                route = [j for k, j in sorted_visits]
                if route:
                    final_routes.append(route)
            
            self.solution = final_routes
    
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
            if current_load > capacities[i]:
                print(f"Error: Vehicle {i} exceeds capacity. Load: {current_load}, Capacity: {capacities[i]}")
                return False

            if not route: continue
            
            current_time = 0.0
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
                    print(f"Error: Vehicle {i} late for stop {to_node}. Arrival: {current_time}, Due: {due_date}")
                    return False
                current_time = max(current_time, ready_time)
                current_time += service_times[to_node]

        # FIX: Added a check to ensure all required customers have been visited.
        # This is the crucial step that was missing.
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

