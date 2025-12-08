import unittest
from unittest.mock import MagicMock, patch
import sys
import os

# --- PATH SETUP ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..', '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)



from graph_coarsening.quantum_solvers.qubo_solver import Qubo
from graph_coarsening.quantum_solvers.vrp_problem import VRPProblem
from graph_coarsening.quantum_solvers.vrp_solution import VRPSolution
from graph_coarsening.quantum_solvers.vrp_solvers import FullQuboSolver, AveragePartitionSolver
import graph_coarsening.quantum_solvers.DWaveSolvers_modified as DWaveSolvers_modified


class TestQuboSolver(unittest.TestCase):
    def test_add_basic(self):
        q = Qubo()
        q.add((1, 2), 5)
        q.add((2, 1), 3) 
        key = (1, 2)
        self.assertIn(key, q.dict)
        self.assertEqual(q.dict[key], 8)

    def test_mixed_type_sorting_fix(self):
        q = Qubo()
        mixed_key = ((0, 1, 0), ('s', 0, 1))
        try:
            q.add(mixed_key, 10)
        except TypeError:
            self.fail("Qubo.add raised TypeError on mixed types. The fix is missing.")
        self.assertEqual(list(q.dict.values())[0], 10)

    def test_constraints(self):
        q = Qubo()
        penalty = 100
        q.add_only_one_constraint(['x', 'y'], penalty)
        self.assertEqual(q.dict[('x', 'x')], -penalty)
        self.assertEqual(q.dict[('y', 'y')], -penalty)
        self.assertEqual(q.dict[('x', 'y')], 2 * penalty)


class TestVRPProblem(unittest.TestCase):
    def setUp(self):
        self.costs = [[0, 10, 10], [10, 0, 10], [10, 10, 0]] 
        self.time_costs = self.costs
        self.capacities = [10, 10]
        self.dests = [1, 2] 
        self.weights = {1: 5, 2: 5}
        self.time_windows = {0: (0, 100), 1: (0, 100), 2: (0, 100)}
        self.service_times = {0: 0, 1: 5, 2: 5}
        
        self.problem = VRPProblem(
            source_depot=0,
            costs=self.costs,
            time_costs=self.time_costs,
            capacities=self.capacities,
            dests=self.dests,
            weights=self.weights,
            time_windows=self.time_windows,
            service_times=self.service_times
        )

    def test_get_qubo_generation(self):
        vehicle_k_limits = [2, 2]
        qubo = self.problem.get_qubo(
            vehicle_k_limits=vehicle_k_limits,
            only_one_const=1000,
            order_const=1,
            capacity_penalty=100,
            time_window_penalty=100,
            vehicle_start_cost=50
        )
        self.assertIsInstance(qubo, Qubo)
        self.assertTrue(len(qubo.dict) > 0)
        self.assertIn(((0, 1, 0), (0, 1, 0)), qubo.dict)


class TestVRPSolution(unittest.TestCase):
    def setUp(self):
        self.problem = MagicMock()
        self.problem.capacities = [10]
        self.problem.source_depot = 0
        self.problem.dests = [1, 2]
        self.problem.weights = {1: 5, 2: 4}
        self.problem.time_windows = {0: (0, 100), 1: (0, 100), 2: (0, 100)}
        self.problem.service_times = {0: 0, 1: 10, 2: 10}
        self.problem.costs = [[0, 5, 5], [5, 0, 5], [5, 5, 0]]

    def test_solution_parsing_and_slack_filtering(self):
        sample = {
            (0, 1, 0): 1,       
            (0, 2, 1): 1,       
            ('s', 0, 0): 1,     
            (0, 1, 1): 0        
        }
        sol = VRPSolution(self.problem, sample, [2])
        self.assertEqual(len(sol.solution), 1) 
        self.assertEqual(sol.solution[0], [1, 2]) 

    def test_validation_missing_customer(self):
        sample = {(0, 1, 0): 1}
        sol = VRPSolution(self.problem, sample, [2])
        with patch('builtins.print'):
            self.assertFalse(sol.check())

    def test_validation_success(self):
        sample = {(0, 1, 0): 1, (0, 2, 1): 1}
        sol = VRPSolution(self.problem, sample, [2])
        self.assertTrue(sol.check())


class TestVRPSolvers(unittest.TestCase):
    def setUp(self):
        self.problem = MagicMock()
        self.problem.dests = [1, 2]
        self.problem.capacities = [10]
        self.problem.source_depot = 0
        self.problem.costs = [[0]*3]*3
        self.problem.get_qubo.return_value = Qubo()

    
    @patch('graph_coarsening.quantum_solvers.DWaveSolvers_modified.solve_qubo')
    def test_full_qubo_solver(self, mock_solve_qubo):
        mock_sample = {(0, 1, 0): 1, (0, 2, 1): 1}
        mock_solve_qubo.return_value = [mock_sample]
        
        solver = FullQuboSolver(self.problem)
        
        solution = solver.solve(
            only_one_const=1, order_const=1, capacity_penalty=1, 
            time_window_penalty=1, vehicle_start_cost=1, 
            solver_type='simulated', num_reads=1
        )
        
        self.assertIsInstance(solution, VRPSolution)
        # If solution is empty, it means the patch didn't work or logic failed
        self.assertEqual(len(solution.solution), 1, "Solution routes list is empty")
        self.assertEqual(solution.solution[0], [1, 2])

    @patch('graph_coarsening.quantum_solvers.DWaveSolvers_modified.solve_qubo')
    def test_average_partition_solver(self, mock_solve_qubo):
        mock_sample = {(0, 1, 0): 1}
        mock_solve_qubo.return_value = [mock_sample]
        
        solver = AveragePartitionSolver(self.problem)
        
        solver.solve(
            only_one_const=1, order_const=1, capacity_penalty=1, 
            time_window_penalty=1, vehicle_start_cost=1, 
            solver_type='simulated', num_reads=1
        )
        
        args, _ = self.problem.get_qubo.call_args
        vehicle_k_limits = args[0]
        self.assertEqual(vehicle_k_limits, [3])

if __name__ == '__main__':
    unittest.main()