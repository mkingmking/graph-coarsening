import math
# You will need to replace this mock with your actual DWaveSolvers_modified import
# from DWaveSolvers_modified import solve_qubo
from mock_dwave_solvers import MockDWaveSolvers as DWaveSolvers_modified # For local testing

from vrp_problem_qubo import VRPProblem
from vrp_solution import VRPSolution

class VRPSolver:
    def __init__(self, problem: VRPProblem):
        self.problem = problem

    def solve(self, only_one_const: float, order_const: float, tw_penalty_const: float, solver_type: str, num_reads: int):
        # This method is meant to be overridden by subclasses
        raise NotImplementedError("Solve method must be implemented by subclasses.")

class FullQuboSolver(VRPSolver):
    def solve(self, only_one_const: float, order_const: float, tw_penalty_const: float, solver_type='simulated', num_reads=50) -> VRPSolution:
        """
        Solves the VRP using a QUBO formulation where each vehicle can potentially
        visit all customers (k_max = num_customers).
        """
        num_customers = len(self.problem.customer_ids)
        num_vehicles = len(self.problem.capacities)
        
        # For FQS, each vehicle can potentially visit all customers.
        # This sets the maximum number of steps a vehicle can take to the total number of customers.
        k_max = num_customers 
        vehicle_k_limits = [k_max] * num_vehicles

        # Generate the QUBO for the VRP problem, including time window penalties
        vrp_qubo = self.problem.get_qubo(vehicle_k_limits, only_one_const, order_const, tw_penalty_const)
        
        # Solve the QUBO using the specified D-Wave solver (or mock)
        # 'limit=1' means we only care about the best sample found
        samples = DWaveSolvers_modified.solve_qubo(vrp_qubo, solver_type=solver_type, limit=1, num_reads=num_reads)
        
        if not samples:
             # If no samples are returned, return an empty solution
             return VRPSolution(self.problem, {}, vehicle_k_limits) # Pass empty sample

        # Create a VRPSolution object from the best sample
        solution = VRPSolution(self.problem, samples[0], vehicle_k_limits)
        return solution

class AveragePartitionSolver(VRPSolver):
    def solve(self, only_one_const: float, order_const: float, tw_penalty_const: float, solver_type='simulated', num_reads=50, limit_radius=1) -> VRPSolution:
        """
        Solves the VRP using a QUBO formulation where the number of steps
        per vehicle is limited based on an average partition heuristic.
        """
        num_customers = len(self.problem.customer_ids)
        num_vehicles = len(self.problem.capacities)

        # For APS, we restrict the number of steps per vehicle.
        # This is a heuristic to reduce the QUBO size by limiting the maximum route length.
        avg_per_vehicle = math.ceil(num_customers / num_vehicles)
        k_max = avg_per_vehicle + limit_radius # Add a small radius for flexibility
        
        vehicle_k_limits = [k_max] * num_vehicles

        # Generate the QUBO for the VRP problem with restricted k_max, including time window penalties
        vrp_qubo = self.problem.get_qubo(vehicle_k_limits, only_one_const, order_const, tw_penalty_const)
        
        # Solve the QUBO using the specified D-Wave solver (or mock)
        samples = DWaveSolvers_modified.solve_qubo(vrp_qubo, solver_type=solver_type, limit=1, num_reads=num_reads)

        if not samples:
             # If no samples are returned, return an empty solution
             return VRPSolution(self.problem, {}, vehicle_k_limits) # Pass empty sample

        # Create a VRPSolution object from the best sample
        solution = VRPSolution(self.problem, samples[0], vehicle_k_limits)
        return solution

