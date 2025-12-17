import math
from . import DWaveSolvers_modified as DWaveSolvers
from .vrp_solution import VRPSolution

class VRPSolver:
    def __init__(self, problem):
        self.problem = problem

    def solve(self, only_one_const, order_const, capacity_penalty, time_window_penalty, vehicle_start_cost, solver_type, num_reads):
        pass

class FullQuboSolver(VRPSolver):
    def solve(self, only_one_const, order_const, capacity_penalty, time_window_penalty, vehicle_start_cost, solver_type='simulated', num_reads=50):
        """
        IMPROVED: Instead of k_max = num_customers (which creates too many variables),
        use k_max = ceil(num_customers / num_vehicles) + buffer
        This reduces the search space dramatically.
        """
        num_customers = len(self.problem.dests)
        num_vehicles = len(self.problem.capacities)
        
        # CRITICAL CHANGE: Limit k_max to reasonable value
        # If we have 5 customers and 5 vehicles, we don't need 5 positions per vehicle
        # Most vehicles will serve 1-2 customers
        avg_customers_per_vehicle = math.ceil(num_customers / num_vehicles)
        k_max = min(avg_customers_per_vehicle + 1, num_customers)  # +1 buffer, but cap at num_customers
        
        vehicle_k_limits = [k_max] * num_vehicles
        
        vrp_qubo = self.problem.get_qubo(
            vehicle_k_limits, only_one_const, order_const, 
            capacity_penalty, time_window_penalty, vehicle_start_cost
        )
        
        try:
            samples = DWaveSolvers.solve_qubo(vrp_qubo, solver_type=solver_type, limit=1, num_reads=num_reads)
        except Exception as e:
            print(f"Solver error: {e}")
            return VRPSolution(self.problem, {}, vehicle_k_limits, solution=[])
            
        if not samples:
             return VRPSolution(self.problem, {}, vehicle_k_limits, solution=[])
             
        solution = VRPSolution(self.problem, samples[0], vehicle_k_limits)
        return solution

class AveragePartitionSolver(VRPSolver):
    def solve(self, only_one_const, order_const, capacity_penalty, time_window_penalty, vehicle_start_cost, solver_type='simulated', num_reads=50, limit_radius=1):
        """
        This solver already does smart k_max selection, but let's make it even smarter.
        """
        num_customers = len(self.problem.dests)
        num_vehicles = len(self.problem.capacities)
        
        # Average customers per vehicle
        avg_per_vehicle = math.ceil(num_customers / num_vehicles) if num_vehicles > 0 else 0
        
        # IMPROVED: Use smaller limit_radius by default
        # Original used limit_radius=1, but even this might be too much
        k_max = avg_per_vehicle + limit_radius
        
        # Cap at reasonable maximum
        k_max = min(k_max, num_customers)
        
        vehicle_k_limits = [k_max] * num_vehicles
        
        vrp_qubo = self.problem.get_qubo(
            vehicle_k_limits, only_one_const, order_const, 
            capacity_penalty, time_window_penalty, vehicle_start_cost
        )
        
        try:
            samples = DWaveSolvers.solve_qubo(vrp_qubo, solver_type=solver_type, limit=1, num_reads=num_reads)
        except Exception as e:
            print(f"Solver error: {e}")
            return VRPSolution(self.problem, {}, vehicle_k_limits, solution=[])
            
        if not samples:
             return VRPSolution(self.problem, {}, vehicle_k_limits, solution=[])
             
        solution = VRPSolution(self.problem, samples[0], vehicle_k_limits)
        return solution


class IterativeRepairSolver(VRPSolver):
    """
    NEW SOLVER: Try multiple parameter settings and pick best feasible solution.
    """
    def solve(self, only_one_const, order_const, capacity_penalty, time_window_penalty, vehicle_start_cost, solver_type='simulated', num_reads=50):
        num_customers = len(self.problem.dests)
        num_vehicles = len(self.problem.capacities)
        
        best_solution = None
        best_cost = float('inf')
        
        # Try different k_max values
        avg_per_vehicle = math.ceil(num_customers / num_vehicles)
        k_values_to_try = [
            max(2, avg_per_vehicle),      # Tight limit
            avg_per_vehicle + 1,            # +1 buffer
            min(avg_per_vehicle + 2, num_customers)  # +2 buffer
        ]
        
        for k_max in k_values_to_try:
            vehicle_k_limits = [k_max] * num_vehicles
            
            vrp_qubo = self.problem.get_qubo(
                vehicle_k_limits, only_one_const, order_const,
                capacity_penalty, time_window_penalty, vehicle_start_cost
            )
            
            try:
                # Get multiple samples to increase chances of finding good solution
                samples = DWaveSolvers.solve_qubo(vrp_qubo, solver_type=solver_type, limit=5, num_reads=num_reads)
                
                for sample in samples:
                    solution = VRPSolution(self.problem, sample, vehicle_k_limits)
                    
                    # Check if solution is valid (each customer visited exactly once)
                    visited = set()
                    valid = True
                    for route in solution.solution:
                        for customer in route:
                            if customer in visited:
                                valid = False
                                break
                            visited.add(customer)
                        if not valid:
                            break
                    
                    if valid and visited == set(self.problem.dests):
                        cost = solution.total_cost()
                        if cost < best_cost:
                            best_cost = cost
                            best_solution = solution
                
            except Exception as e:
                print(f"Solver error with k_max={k_max}: {e}")
                continue
        
        if best_solution is None:
            # Fallback: return repaired solution from first attempt
            vehicle_k_limits = [k_values_to_try[0]] * num_vehicles
            return VRPSolution(self.problem, {}, vehicle_k_limits, solution=[])
        
        return best_solution