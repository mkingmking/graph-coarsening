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
                if val == 1:
                    i, j, k = var
                    temp_routes[i].append((k, j))

            final_routes = []
            for i in range(num_vehicles):
                sorted_visits = sorted(temp_routes[i], key=lambda x: x[0])
                route = [j for k, j in sorted_visits]
                final_routes.append(route)
            
            self.solution = final_routes

    def check(self):
        capacities = self.problem.capacities
        weights = self.problem.weights
        time_windows = self.problem.time_windows
        service_times = self.problem.service_times

        # --- Check for duplicate visits ---
        all_visited_dests_list = [dest for route in self.solution for dest in route]
        
        # 1. Did we visit the correct number of locations?
        if len(all_visited_dests_list) != len(self.problem.dests):
            print(f"Warning: Incorrect number of total visits. Required: {len(self.problem.dests)}, Found: {len(all_visited_dests_list)}")
            return False
            
        # 2. Are there any duplicate visits across all routes?
        if len(set(all_visited_dests_list)) != len(all_visited_dests_list):
            print(f"Warning: Duplicate customer visits found in solution.")
            return False

        # 3. Capacity Check
        for i, route in enumerate(self.solution):
            vehicle_load = sum(weights[dest] for dest in route)
            if vehicle_load > capacities[i]:
                print(f"Warning: Vehicle {i} exceeds capacity. Load: {vehicle_load}, Capacity: {capacities[i]}")
                return False

        # 4. Time Window Check
        costs = self.problem.costs
        for i, route in enumerate(self.solution):
            if not route:
                continue

            current_time = 0
            
            # Travel from depot to first customer
            current_time += costs[self.depot][route[0]]
            
            # Check time window for first customer
            ready_time, due_date = time_windows[route[0]]
            if current_time > due_date:
                print(f"Warning: Vehicle {i} arrives at customer {route[0]} too late. Arrival: {current_time}, Due Date: {due_date}")
                return False
            
            # Wait if necessary
            current_time = max(current_time, ready_time)
            
            # Service time at first customer
            current_time += service_times[route[0]]
            
            for j in range(len(route) - 1):
                # Travel from current customer to next customer
                current_time += costs[route[j]][route[j+1]]
                
                # Check time window for next customer
                ready_time, due_date = time_windows[route[j+1]]
                if current_time > due_date:
                    print(f"Warning: Vehicle {i} arrives at customer {route[j+1]} too late. Arrival: {current_time}, Due Date: {due_date}")
                    return False
                
                # Wait if necessary
                current_time = max(current_time, ready_time)
                
                # Service time at next customer
                current_time += service_times[route[j+1]]

        return True

    def total_cost(self):
        costs = self.problem.costs
        depot = self.depot
        total_cost = 0

        for route in self.solution:
            if not route:
                continue
            
            # Calculate total travel time and waiting time
            current_time = 0
            total_waiting_time = 0
            
            # Travel from depot to first customer
            travel_time_first = costs[depot][route[0]]
            arrival_time_first = current_time + travel_time_first
            
            # Waiting time at first customer
            waiting_time_first = max(0, self.problem.time_windows[route[0]][0] - arrival_time_first)
            
            # Update current time after service
            current_time = max(arrival_time_first, self.problem.time_windows[route[0]][0]) + self.problem.service_times[route[0]]
            
            total_waiting_time += waiting_time_first
            
            current_cost = travel_time_first
            
            for i in range(len(route) - 1):
                travel_time = costs[route[i]][route[i+1]]
                current_cost += travel_time
                
                arrival_time = current_time + travel_time
                waiting_time = max(0, self.problem.time_windows[route[i+1]][0] - arrival_time)
                
                current_time = max(arrival_time, self.problem.time_windows[route[i+1]][0]) + self.problem.service_times[route[i+1]]
                
                total_waiting_time += waiting_time

            # Return to depot
            current_cost += costs[route[-1]][depot]
            
            total_cost += current_cost + total_waiting_time
            
        return total_cost

    def description(self):
        print("Solution Routes:")
        for i, route in enumerate(self.solution):
            if route:
                full_path = [self.depot] + route + [self.depot]
                print(f"  Vehicle {i}: {' -> '.join(map(str, full_path))}")
            else:
                print(f"  Vehicle {i}: Not used")
        print(f"\nTotal Cost: {self.total_cost():.2f}")
        print(f"Is Solution Valid: {self.check()}")