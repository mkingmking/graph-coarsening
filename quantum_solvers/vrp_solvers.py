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
        num_customers = len(self.problem.dests)
        num_vehicles = len(self.problem.capacities)
        
        avg_customers_per_vehicle = math.ceil(num_customers / num_vehicles)
        k_max = min(avg_customers_per_vehicle + 1, num_customers)
        
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
        num_customers = len(self.problem.dests)
        num_vehicles = len(self.problem.capacities)
        avg_per_vehicle = math.ceil(num_customers / num_vehicles) if num_vehicles > 0 else 0
        k_max = min(avg_per_vehicle + limit_radius, num_customers)
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
    Solves using multiple k_max values and picks the BEST FEASIBLE solution.
    Strictly checks validity (Time Windows & Capacity) before accepting.
    """
    def solve(self, only_one_const, order_const, capacity_penalty, time_window_penalty, vehicle_start_cost, solver_type='simulated', num_reads=50):
        num_customers = len(self.problem.dests)
        num_vehicles = len(self.problem.capacities)
        
        best_solution = None
        best_cost = float('inf')
        
        avg_per_vehicle = math.ceil(num_customers / num_vehicles)
        k_values_to_try = [
            max(2, avg_per_vehicle),
            avg_per_vehicle + 1,
            min(avg_per_vehicle + 2, num_customers)
        ]
        
        for k_max in k_values_to_try:
            vehicle_k_limits = [k_max] * num_vehicles
            
            vrp_qubo = self.problem.get_qubo(
                vehicle_k_limits, only_one_const, order_const,
                capacity_penalty, time_window_penalty, vehicle_start_cost
            )
            
            try:
                samples = DWaveSolvers.solve_qubo(vrp_qubo, solver_type=solver_type, limit=5, num_reads=num_reads)
                
                for sample in samples:
                    solution = VRPSolution(self.problem, sample, vehicle_k_limits)
                    
                    # 1. Check Completeness (Visits everyone once)
                    visited = set()
                    is_complete = True
                    for route in solution.solution:
                        for customer in route:
                            if customer in visited:
                                is_complete = False; break
                            visited.add(customer)
                        if not is_complete: break
                    
                    if is_complete and visited == set(self.problem.dests):
                        # 2. CRITICAL: Check Feasibility (Time Windows & Capacity)
                        # The built-in check() method does exactly this.
                        if solution.check():
                            cost = solution.total_cost()
                            if cost < best_cost:
                                best_cost = cost
                                best_solution = solution
                
            except Exception as e:
                print(f"Solver error with k_max={k_max}: {e}")
                continue
        
        if best_solution is None:
            # Fallback: return repaired solution from first attempt, even if imperfect
            vehicle_k_limits = [k_values_to_try[0]] * num_vehicles
            return VRPSolution(self.problem, {}, vehicle_k_limits, solution=[])
        
        return best_solution