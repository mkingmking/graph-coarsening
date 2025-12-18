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
        
        # PRE-CALCULATION: True Earliest Possible Arrival Times
        # No matter where you come from, you cannot arrive at J earlier than
        # driving straight from the Depot.
        self.true_earliest = {}
        # Assume depot start time is the start of its window (usually 0)
        depot_start = self.time_windows[self.source_depot][0]
        
        for j in self.dests:
            travel_from_depot = self.time_costs[self.source_depot][j]
            # You arrive at J either when it opens OR when you get there from depot
            # whichever is LATER.
            arrival_limit = max(self.time_windows[j][0], depot_start + travel_from_depot)
            self.true_earliest[j] = arrival_limit

    def get_qubo(self, vehicle_k_limits, only_one_const, order_const, capacity_penalty, time_window_penalty, vehicle_start_cost):
        """
        Generates the QUBO for the CVRPTW with PHYSICS-AWARE TIME CONSTRAINTS.
        """
        num_vehicles = len(self.capacities)
        customer_nodes = self.dests
        
        qubo = Qubo()

        # =================================================================
        # 1. UNIQUE VISIT CONSTRAINTS (Standard)
        # =================================================================
        penalty_scale = only_one_const
        
        # Each customer visited exactly once
        for j in customer_nodes:
            variables = [(i, j, k) for i in range(num_vehicles) for k in range(vehicle_k_limits[i])]
            for var in variables:
                qubo.add((var, var), -penalty_scale)
            for i in range(len(variables)):
                for j_idx in range(i + 1, len(variables)):
                    qubo.add((variables[i], variables[j_idx]), 2 * penalty_scale)

        # Each vehicle at most one customer per step
        for i in range(num_vehicles):
            for k in range(vehicle_k_limits[i]):
                variables = [(i, j, k) for j in customer_nodes]
                for idx1 in range(len(variables)):
                    for idx2 in range(idx1 + 1, len(variables)):
                        qubo.add((variables[idx1], variables[idx2]), penalty_scale)

        # =================================================================
        # 2. ROUTE CONTINUITY & SELF-LOOPS
        # =================================================================
        continuity_penalty = only_one_const * 0.1
        selfloop_penalty = only_one_const * 0.5
        
        for i in range(num_vehicles):
            # Continuity: Penalize gaps (using k without k-1)
            for k in range(1, vehicle_k_limits[i]):
                for j in customer_nodes:
                    var_k = (i, j, k)
                    qubo.add((var_k, var_k), continuity_penalty)
                    for j_prev in customer_nodes:
                        var_k_prev = (i, j_prev, k - 1)
                        qubo.add((var_k, var_k_prev), -continuity_penalty * 0.5)

            # Self-loops: Cannot visit same node twice in a row
            for k in range(vehicle_k_limits[i] - 1):
                for j in customer_nodes:
                    qubo.add(((i, j, k), (i, j, k + 1)), selfloop_penalty)

        # =================================================================
        # 3. CAPACITY CONSTRAINT (Logarithmic Slack)
        # =================================================================
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

        # =================================================================
        # 4. PHYSICS-AWARE TIME WINDOW CONSTRAINTS
        # =================================================================
        
        # A. DEPOT INITIAL CHECK
        # If a customer is so far that even driving straight from depot makes them late
        for i in range(num_vehicles):
            for j1 in customer_nodes:
                if self.true_earliest[j1] > self.time_windows[j1][1]:
                    qubo.add(((i, j1, 0), (i, j1, 0)), time_window_penalty)

        # B. PAIRWISE CHECK (Step k -> Step k+1)
        # Using TRUE EARLIEST times instead of naive window open times
        for i in range(num_vehicles):
            k_max = vehicle_k_limits[i]
            for j1 in customer_nodes:
                for j2 in customer_nodes:
                    if j1 == j2: continue
                    
                    # TRUE EARLIEST LEAVE TIME
                    # We use the pre-calculated strict lower bound
                    # arrival_at_j1 >= self.true_earliest[j1]
                    earliest_leave_j1 = self.true_earliest[j1] + self.service_times[j1]
                    earliest_arrival_j2 = earliest_leave_j1 + self.time_costs[j1][j2]
                    
                    # 1. HARD CHECK
                    if earliest_arrival_j2 > self.time_windows[j2][1]:
                        # This link is physically impossible
                        for k in range(k_max - 1):
                            qubo.add(((i, j1, k), (i, j2, k + 1)), time_window_penalty)
                    
                    # 2. RISK CHECK (Soft Constraint)
                    # Even if possible, if it's tight, penalize it to avoid accumulated error
                    elif earliest_arrival_j2 > self.time_windows[j2][1] * 0.9: 
                        risk_penalty = time_window_penalty * 0.05 
                        for k in range(k_max - 1):
                            qubo.add(((i, j1, k), (i, j2, k + 1)), risk_penalty)

        # C. TRIANGLE LOOKAHEAD (Step k -> Step k+2)
        # Stricter version using True Earliest
        for i in range(num_vehicles):
            k_max = vehicle_k_limits[i]
            if k_max < 3: continue 
            
            for j1 in customer_nodes:      
                for j3 in customer_nodes:  
                    if j1 == j3: continue
                    
                    earliest_leave_j1 = self.true_earliest[j1] + self.service_times[j1]
                    
                    # Can we bridge j1 -> j2 -> j3?
                    possible_connection = False
                    for j2 in customer_nodes:
                        if j2 == j1 or j2 == j3: continue
                        
                        arrival_j2 = earliest_leave_j1 + self.time_costs[j1][j2]
                        if arrival_j2 > self.time_windows[j2][1]: continue # Can't reach mid
                        
                        # Wait at j2 if early
                        leave_j2 = max(arrival_j2, self.true_earliest[j2]) + self.service_times[j2]
                        arrival_j3 = leave_j2 + self.time_costs[j2][j3]
                        
                        if arrival_j3 <= self.time_windows[j3][1]:
                            possible_connection = True
                            break 
                    
                    if not possible_connection:
                        for k in range(k_max - 2):
                            qubo.add(((i, j1, k), (i, j3, k + 2)), time_window_penalty)

        # =================================================================
        # 5. OBJECTIVE FUNCTION (Clarke-Wright Savings)
        # =================================================================
        for i in range(num_vehicles):
            k_max = vehicle_k_limits[i]
            
            # Linear Terms: Base cost (Round trip assumption)
            for k in range(k_max):
                for j in customer_nodes:
                    cost_round_trip = self.costs[self.source_depot][j] + self.costs[j][self.source_depot]
                    penalty_val = cost_round_trip * order_const
                    if k == 0:
                        penalty_val += vehicle_start_cost
                    qubo.add(((i, j, k), (i, j, k)), penalty_val)

            # Quadratic Terms: Savings (Connecting j1 -> j2 saves return trip)
            for k in range(k_max - 1):
                for j1 in customer_nodes:
                    for j2 in customer_nodes:
                        if j1 == j2: continue
                        var1 = (i, j1, k)
                        var2 = (i, j2, k + 1)
                        savings_cost = (self.costs[j1][j2] 
                                      - self.costs[j1][self.source_depot] 
                                      - self.costs[self.source_depot][j2])
                        qubo.add((var1, var2), savings_cost * order_const)

        return qubo