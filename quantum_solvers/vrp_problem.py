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
        Generates the QUBO for the CVRPTW with MULTI-STEP TIME LOOKAHEAD.
        """
        num_vehicles = len(self.capacities)
        customer_nodes = self.dests
        
        qubo = Qubo()

       
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
        # ADVANCED TIME WINDOW CONSTRAINTS
        # =================================================================
        
        # A. PAIRWISE CHECK (Step k -> Step k+1)
        for i in range(num_vehicles):
            k_max = vehicle_k_limits[i]
            
            # Depot -> First Customer
            for j1 in customer_nodes:
                # Earliest Arrival
                depot_ready = self.time_windows[self.source_depot][0]
                earliest_arrival = depot_ready + self.service_times[self.source_depot] + self.time_costs[self.source_depot][j1]
                
                if earliest_arrival > self.time_windows[j1][1]:
                    qubo.add(((i, j1, 0), (i, j1, 0)), time_window_penalty)

            # Customer -> Customer
            for j1 in customer_nodes:
                for j2 in customer_nodes:
                    if j1 == j2: continue
                    
                    # 1. OPTIMISTIC CHECK (Hard Constraint)
                    
                    earliest_leave_j1 = self.time_windows[j1][0] + self.service_times[j1]
                    earliest_arrival_j2 = earliest_leave_j1 + self.time_costs[j1][j2]
                    
                    if earliest_arrival_j2 > self.time_windows[j2][1]:
                        for k in range(k_max - 1):
                            qubo.add(((i, j1, k), (i, j2, k + 1)), time_window_penalty)
                            
                    # 2. PESSIMISTIC "RISK" CHECK (Soft Constraint)
                    
                    latest_leave_j1 = self.time_windows[j1][1] + self.service_times[j1]
                    latest_arrival_j2 = latest_leave_j1 + self.time_costs[j1][j2]
                    
                    if latest_arrival_j2 > self.time_windows[j2][1]:
                        risk_penalty = time_window_penalty * 0.1  # 10% of full penalty
                        for k in range(k_max - 1):
                            qubo.add(((i, j1, k), (i, j2, k + 1)), risk_penalty)

        # B. TRIANGLE LOOKAHEAD CHECK (Step k -> Step k+2)
        # Checks A -> B -> C to catch accumulated delays.
        for i in range(num_vehicles):
            k_max = vehicle_k_limits[i]
            if k_max < 3: continue  # Need at least 3 steps for A->B->C
            
            for j1 in customer_nodes:      # Node at k
                for j3 in customer_nodes:  # Node at k+2
                    if j1 == j3: continue
                    
                    
                    min_time_j1_to_j3 = float('inf')
                    
                    earliest_leave_j1 = self.time_windows[j1][0] + self.service_times[j1]
                    
                    # Scan all possible middle nodes
                    possible_connection = False
                    for j2 in customer_nodes:
                        if j2 == j1 or j2 == j3: continue
                        
                        arrival_j2 = earliest_leave_j1 + self.time_costs[j1][j2]
                        
                        # If j2 is unreachable from j1, skip
                        if arrival_j2 > self.time_windows[j2][1]:
                            continue
                            
                        # Wait at j2 if early
                        leave_j2 = max(arrival_j2, self.time_windows[j2][0]) + self.service_times[j2]
                        arrival_j3 = leave_j2 + self.time_costs[j2][j3]
                        
                        if arrival_j3 <= self.time_windows[j3][1]:
                            possible_connection = True
                            break # Found at least one valid middle man
                    
                    # If NO node j2 can bridge j1 and j3 in time:
                    if not possible_connection:
                        # Penalize having j1 at k and j3 at k+2 simultaneously
                        for k in range(k_max - 2):
                            qubo.add(((i, j1, k), (i, j3, k + 2)), time_window_penalty)

        # =================================================================
        #  OBJECTIVE FUNCTION 
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
                        
                        # Savings = Cost(j1, j2) - Cost(j1, Depot) - Cost(Depot, j2)
                        savings_cost = (self.costs[j1][j2] 
                                      - self.costs[j1][self.source_depot] 
                                      - self.costs[self.source_depot][j2])
                        
                        qubo.add((var1, var2), savings_cost * order_const)

        return qubo