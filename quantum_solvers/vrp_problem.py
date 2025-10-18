import math
from .qubo_solver import Qubo

class VRPProblem:
    def __init__(self, source_depot, costs, time_costs, capacities, dests, weights, time_windows, service_times):
        self.source_depot = source_depot
        self.costs = costs
        self.time_costs = time_costs
        self.capacities = capacities
        self.dests = dests
        self.weights = weights
        self.time_windows = time_windows
        self.service_times = service_times

    def get_qubo(self, vehicle_k_limits, only_one_const, order_const, capacity_penalty, time_window_penalty, vehicle_start_cost):
        """
        Generates the QUBO for the CVRPTW.
        This model now includes penalties for vehicle capacity and time window violations.
        """
        num_vehicles = len(self.capacities)
        customer_nodes = self.dests

        qubo = Qubo()

        # --- Base Constraints ---
        # 1. Each customer is visited exactly once (globally).
        for j in customer_nodes:
            variables = [(i, j, k) for i in range(num_vehicles) for k in range(vehicle_k_limits[i])]
            qubo.add_only_one_constraint(variables, only_one_const)

        # 2. Each vehicle is in at most one location at each step.
        for i in range(num_vehicles):
            for k in range(vehicle_k_limits[i]):
                variables = [(i, j, k) for j in customer_nodes]
                qubo.add_at_most_one_constraint(variables, only_one_const)
        
        # --- NEW: A vehicle cannot visit the same customer more than once. ---
        # This is a critical addition to prevent routes like (C4 -> C4).
        for i in range(num_vehicles):
            for j in customer_nodes:
                # For a given vehicle 'i' and customer 'j', only one 'k' (step) can be chosen.
                variables = [(i, j, k) for k in range(vehicle_k_limits[i])]
                qubo.add_at_most_one_constraint(variables, only_one_const)


        # --- Capacity Constraint ---
        # For each vehicle, sum of demands + slack variables must equal capacity.
        for i in range(num_vehicles):
            capacity = self.capacities[i]
            if capacity <= 0: continue
            
            num_slack_bits = math.floor(math.log2(capacity)) + 1
            slack_vars = [('s', i, m) for m in range(num_slack_bits)]
            
            constraint_expr = []
            for j in customer_nodes:
                demand = self.weights.get(j, 0)
                for k in range(vehicle_k_limits[i]):
                    constraint_expr.append((demand, (i, j, k)))
            for m in range(num_slack_bits):
                constraint_expr.append((2**m, slack_vars[m]))
            
            qubo.add_quadratic_equality_constraint(constraint_expr, -capacity, capacity_penalty)

        # --- Time Window Constraint ---
        # Penalize transitions between customers that are impossible due to time windows.
        for i in range(num_vehicles):
            k_max = vehicle_k_limits[i]
            for j1 in customer_nodes:
                depot_ready_time = self.time_windows[self.source_depot][0]
                depot_service_time = self.service_times[self.source_depot]
                travel_time_to_j1 = self.time_costs[self.source_depot][j1]
                earliest_arrival_j1 = depot_ready_time + depot_service_time + travel_time_to_j1
                if earliest_arrival_j1 > self.time_windows[j1][1]:
                    qubo.add(((i, j1, 0), (i, j1, 0)), time_window_penalty)
                
                for j2 in customer_nodes:
                    if j1 == j2: continue
                    earliest_finish_j1 = self.time_windows[j1][0] + self.service_times[j1]
                    travel_time_j1_j2 = self.time_costs[j1][j2]
                    earliest_arrival_j2 = earliest_finish_j1 + travel_time_j1_j2
                    
                    if earliest_arrival_j2 > self.time_windows[j2][1]:
                        for k in range(k_max - 1):
                            var1 = (i, j1, k)
                            var2 = (i, j2, k + 1)
                            qubo.add((var1, var2), time_window_penalty)

        # --- Objective Function ---
        # Minimize total travel distance + vehicle usage cost.
        for i in range(num_vehicles):
            k_max = vehicle_k_limits[i]
            for j in customer_nodes:
                cost_from_depot = self.costs[self.source_depot][j]
                qubo.add(((i, j, 0), (i, j, 0)), cost_from_depot * order_const + vehicle_start_cost)
                
                cost_to_depot = self.costs[j][self.source_depot]
                qubo.add(((i, j, k_max - 1), (i, j, k_max - 1)), cost_to_depot * order_const)

            for k in range(k_max - 1):
                for j1 in customer_nodes:
                    for j2 in customer_nodes:
                        if j1 == j2: continue
                        var1 = (i, j1, k)
                        var2 = (i, j2, k + 1)
                        cost = self.costs[j1][j2]
                        qubo.add((var1, var2), cost * order_const)

        return qubo

