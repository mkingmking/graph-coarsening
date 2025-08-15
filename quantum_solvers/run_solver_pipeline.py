
import os
import logging
import random
import numpy as np
import math
import csv
from itertools import product
from multiprocessing import Process

# Assuming these files are present in the same directory and are compatible
from graph import Graph, compute_euclidean_tau
from node import Node
from utils import calculate_route_metrics # The parser from utils.py is not used for CVRPTW
from greedy_solver import GreedySolver
from savings_solver import SavingsSolver
from coarsener import SpatioTemporalGraphCoarsener
# Mock D-Wave solver for local testing
import DWaveSolvers_modified as DWaveSolvers
from visualisation import visualize_routes
from vrp_problem import VRPProblem
from vrp_solution import VRPSolution



# --- FullQuboSolver Class ---
class FullQuboSolver(VRPSolver):
    def solve(self, only_one_const, order_const, time_window_const, solver_type='simulated', num_reads=50):
        num_customers = len(self.problem.dests)
        num_vehicles = len(self.problem.capacities)
        k_max = num_customers
        vehicle_k_limits = [k_max] * num_vehicles
        vrp_qubo = self.problem.get_qubo(vehicle_k_limits, only_one_const, order_const, time_window_const)
        try:
            samples = DWaveSolvers.solve_qubo(
                vrp_qubo, 
                solver_type=solver_type, 
                limit=1, 
                num_reads=num_reads
            )
        except Exception as e:
            print(f"Solver error: {e}")
            return VRPSolution(self.problem, {}, vehicle_k_limits, solution=[])
        if not samples:
             return VRPSolution(self.problem, {}, vehicle_k_limits, solution=[])
        solution = VRPSolution(self.problem, samples[0], vehicle_k_limits)
        return solution

# --- AveragePartitionSolver Class ---
class AveragePartitionSolver(VRPSolver):
    def solve(self, only_one_const, order_const, time_window_const, solver_type='simulated', num_reads=50, limit_radius=1):
        num_customers = len(self.problem.dests)
        num_vehicles = len(self.problem.capacities)
        avg_per_vehicle = math.ceil(num_customers / num_vehicles)
        k_max = avg_per_vehicle + limit_radius
        vehicle_k_limits = [k_max] * num_vehicles
        vrp_qubo = self.problem.get_qubo(vehicle_k_limits, only_one_const, order_const, time_window_const)
        try:
            samples = DWaveSolvers.solve_qubo(
                vrp_qubo, 
                solver_type=solver_type, 
                limit=1, 
                num_reads=num_reads
            )
        except Exception as e:
            print(f"Solver error: {e}")
            return VRPSolution(self.problem, {}, vehicle_k_limits, solution=[])
        if not samples:
             return VRPSolution(self.problem, {}, vehicle_k_limits, solution=[])
        solution = VRPSolution(self.problem, samples[0], vehicle_k_limits)
        return solution



  







def run_solver_pipeline_cvrptw(file_path: str, num_customers: int, solver_name: str):
    """
    Runs the CVRPTW solver pipeline using the specified solver.
    """
    print(f"\n--- Running CVRPTW Solver: {solver_name} ---")
    
    # Use the dedicated Solomon parser for CVRPTW data
    data = read_solomon(file_path, num_customers)

    problem = VRPProblem(
        sources=data['sources'],
        costs=data['costs'],
        time_costs=data['time_costs'],
        capacities=data['capacities'],
        dests=data['dests'],
        weights=data['weights'],
        time_windows=data['time_windows'],
        service_times=data['service_times']
    )

    # QUBO parameters for the CVRPTW formulation
    # These constants can be adjusted for fine-tuning
    qubo_params = dict(only_one_const=1000, order_const=1, time_window_const=1000, solver_type='simulated', num_reads=10)

    if solver_name == 'FQS':
        solver = FullQuboSolver(problem)
        solution = solver.solve(**qubo_params)
    elif solver_name == 'APS':
        solver = AveragePartitionSolver(problem)
        solution = solver.solve(**qubo_params)
    else:
        print(f"Error: Unknown solver name '{solver_name}'. Choose 'FQS' or 'APS'.")
        return

    if solution:
        solution.description()
    else:
        print(f"Solver {solver_name} failed to find a solution.")