from .qubo_solver import Qubo
from itertools import product
import numpy as np

class VRPProblem:
    def __init__(self, sources, costs, time_costs, capacities, dests, weights, time_windows, service_times):
        self.source_depot = 0
        self.costs = costs
        self.time_costs = time_costs
        self.capacities = capacities
        self.dests = dests
        self.weights = weights
        self.time_windows = time_windows
        self.service_times = service_times

    def get_qubo(self, vehicle_k_limits, only_one_const, order_const, time_window_const):
        """
        Generates the QUBO for the CVRPTW.
        Variables are tuples (i, j, k): vehicle i, destination j, step k.
        """
        num_vehicles = len(self.capacities)
        customer_nodes = self.dests
        num_nodes = len(self.weights)

        qubo = Qubo()

        # ======================================================================
        # CONSTRAINT 1: Each customer is visited exactly once.
        # ======================================================================
        for j in customer_nodes:
            variables_for_dest_j = []
            for i in range(num_vehicles):
                k_max = vehicle_k_limits[i]
                for k in range(1, k_max + 1):
                    variables_for_dest_j.append((i, j, k))
            qubo.add_only_one_constraint(variables_for_dest_j, only_one_const)

        # ======================================================================
        # CONSTRAINT 2: Each vehicle is in at most one location at each step.
        # ======================================================================
        for i in range(num_vehicles):
            k_max = vehicle_k_limits[i]
            for k in range(1, k_max + 1):
                variables_for_vehicle_step = [(i, j, k) for j in customer_nodes]
                qubo.add_only_one_constraint(variables_for_vehicle_step, only_one_const)
   
        # ======================================================================
        # CONSTRAINT 3: A vehicle cannot visit the same customer twice.
        # ======================================================================
        for i in range(num_vehicles):
            k_max = vehicle_k_limits[i]
            for j in customer_nodes:
                variables_for_vehicle_dest = [(i, j, k) for k in range(1, k_max + 1)]
                for k1_idx in range(len(variables_for_vehicle_dest)):
                    for k2_idx in range(k1_idx + 1, len(variables_for_vehicle_dest)):
                        var1 = variables_for_vehicle_dest[k1_idx]
                        var2 = variables_for_vehicle_dest[k2_idx]
                        qubo.add((var1, var2), only_one_const)

        # ======================================================================
        # OBJECTIVE FUNCTION C: Minimize travel distance BETWEEN CUSTOMERS.
        # ======================================================================
        for i in range(num_vehicles):
            k_max = vehicle_k_limits[i]
            # --- Cost between intermediate stops (step k to k+1) ---
            for k in range(1, k_max): # from step 1 up to k_max-1
                for j1 in customer_nodes:
                    for j2 in customer_nodes:
                        if j1 == j2: continue
                        var1 = (i, j1, k)
                        var2 = (i, j2, k+1)
                        cost = self.costs[j1][j2]
                        qubo.add((var1, var2), cost * order_const)

        # ======================================================================
        # NEW CONSTRAINT: Time Window Constraint
        # This is a soft constraint. It penalizes solutions that violate time windows.
        # Note: The QUBO formulation of this constraint can be complex and may require
        # a different approach, e.g., a time-expanded network.
        # Here, we will formulate a simplified penalty based on a single step.
        # ======================================================================
        for i in range(num_vehicles):
            k_max = vehicle_k_limits[i]
            for k in range(1, k_max + 1):
                # The total time accumulated up to step k
                # This is a simplified approximation and a more rigorous approach
                # would be needed for a precise QUBO formulation.
                # However, for a simple penalty, we can use the k value as a proxy for time.
                for j in customer_nodes:
                    arrival_time = self.costs[self.source_depot][j] * k # a very rough approximation
                    
                    ready_time = self.time_windows[j][0]
                    due_date = self.time_windows[j][1]
                    
                    if arrival_time < ready_time:
                        penalty = (ready_time - arrival_time)
                        qubo.add((i, j, k), penalty * time_window_const)
                    elif arrival_time > due_date:
                        penalty = (arrival_time - due_date)
                        qubo.add((i, j, k), penalty * time_window_const)
        
        return qubo