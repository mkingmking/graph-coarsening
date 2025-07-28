import pytest
import math

from quantum_solvers.vrp_solvers import FullQuboSolver, AveragePartitionSolver, DWaveSolvers_modified
from quantum_solvers.vrp_solution import VRPSolution
from quantum_solvers.vrp_problem_qubo import VRPProblem

# Helper classes for testing
class FakeProblem:
    def __init__(self, customer_ids, capacities):
        self.customer_ids = customer_ids
        self.capacities = capacities
        self.get_qubo_called = False
        self.last_args = None

    def get_qubo(self, vehicle_k_limits, only_one_const, order_const, tw_penalty_const):
        # Record invocation
        self.get_qubo_called = True
        self.last_args = (vehicle_k_limits, only_one_const, order_const, tw_penalty_const)
        # Return dummy placeholder
        class DummyQubo:
            def __init__(self): self.dict = {}
        return DummyQubo()

# DummyNode for VRPSolution tests
class DummyNode:
    def __init__(self, e, s, l):
        self.e = e  # earliest service
        self.s = s  # service duration
        self.l = l  # latest service

# ------------------- FullQuboSolver Tests -------------------

def test_full_qubo_solver_calls_get_qubo_and_returns_solution(monkeypatch):
    problem = FakeProblem(['c1'], [10])
    solver = FullQuboSolver(problem)

    def fake_solve_qubo(qubo, solver_type, limit, num_reads):
        assert solver_type == 'sim'
        assert limit == 1
        assert num_reads == 3
        return [ {(0, 'c1', 1): 1} ]
    monkeypatch.setattr(DWaveSolvers_modified, 'solve_qubo', fake_solve_qubo)

    sol = solver.solve(only_one_const=0.5, order_const=0.6, tw_penalty_const=0.7,
                       solver_type='sim', num_reads=3)
    assert problem.get_qubo_called
    assert problem.last_args == ([1], 0.5, 0.6, 0.7)
    assert isinstance(sol, VRPSolution)
    assert sol.vehicle_k_limits == [1]
    assert sol.solution == [['c1']]


def test_full_qubo_solver_empty_samples(monkeypatch):
    problem = FakeProblem(['c1','c2'], [10,10])
    solver = FullQuboSolver(problem)
    monkeypatch.setattr(DWaveSolvers_modified, 'solve_qubo', lambda *args, **kwargs: [])

    sol = solver.solve(only_one_const=1, order_const=2, tw_penalty_const=3,
                       solver_type='any', num_reads=1)
    # k_max per vehicle equals number of customers (2)
    assert sol.vehicle_k_limits == [2,2]
    # Each vehicle route empty
    assert sol.solution == [[], []]


def test_full_qubo_solver_multi_vehicle(monkeypatch):
    # Three customers, three vehicles
    problem = FakeProblem(['c1','c2','c3'], [10,10,10])
    solver = FullQuboSolver(problem)
    def fake_solve_qubo(qubo, solver_type, limit, num_reads):
        # Assign each vehicle one customer at step 1
        return [{(0,'c1',1):1,(1,'c2',1):1,(2,'c3',1):1}]
    monkeypatch.setattr(DWaveSolvers_modified, 'solve_qubo', fake_solve_qubo)

    sol = solver.solve(only_one_const=1, order_const=1, tw_penalty_const=1,
                       solver_type='sim2', num_reads=1)
    assert problem.get_qubo_called
    assert problem.last_args == ([3,3,3],1,1,1)
    assert sol.vehicle_k_limits == [3,3,3]
    assert sol.solution == [['c1'],['c2'],['c3']]

# ------------------- AveragePartitionSolver Tests -------------------


def test_average_partition_custom_limit_radius(monkeypatch):
    # 3 customers, 2 vehicles, limit_radius=2
    problem = FakeProblem(['c1','c2','c3'], [10,10])
    solver = AveragePartitionSolver(problem)
    def fake_solve_qubo(qubo, solver_type, limit, num_reads):
        return [{(0,'c1',1):1,(1,'c2',1):1}]
    monkeypatch.setattr(DWaveSolvers_modified, 'solve_qubo', fake_solve_qubo)

    sol = solver.solve(only_one_const=2, order_const=2, tw_penalty_const=2,
                       solver_type='local', num_reads=5, limit_radius=2)
    avg = math.ceil(3/2)
    expected_k = avg + 2
    assert problem.get_qubo_called
    assert problem.last_args == ([expected_k, expected_k],2,2,2)
    assert sol.vehicle_k_limits == [expected_k, expected_k]
    assert sol.solution == [['c1'],['c2']]


def test_average_partition_default_limit_radius(monkeypatch):
    # 4 customers, 2 vehicles, default limit_radius=1 (from solver class)
    problem = FakeProblem(['c1','c2','c3','c4'], [10,10])
    solver = AveragePartitionSolver(problem)
    def fake_solve_qubo(qubo, solver_type, limit, num_reads):
        # Distribute two customers per vehicle
        return [{(0,'c1',1):1,(0,'c2',2):1,(1,'c3',1):1,(1,'c4',2):1}]
    monkeypatch.setattr(DWaveSolvers_modified, 'solve_qubo', fake_solve_qubo)
    
    sol = solver.solve(only_one_const=1, order_const=1, tw_penalty_const=1,
                        solver_type='sim', num_reads=1)
    
    num_customers = len(problem.customer_ids)
    num_vehicles = len(problem.capacities)
    avg_per_vehicle = math.ceil(num_customers / num_vehicles) # 4/2 = 2
    default_limit_radius = 1 # Default in AveragePartitionSolver.solve
    expected_k_max = avg_per_vehicle + default_limit_radius # 2 + 1 = 3

    assert problem.last_args == ([expected_k_max, expected_k_max], 1, 1, 1)


def test_average_partition_large_limit_radius(monkeypatch):
    # Verify behavior with large limit_radius
    problem = FakeProblem(['c1','c2','c3'], [10,10])
    solver = AveragePartitionSolver(problem)
    # Corrected lambda to accept all expected arguments
    monkeypatch.setattr(DWaveSolvers_modified, 'solve_qubo', 
                        lambda qubo, solver_type, limit, num_reads: [{ }])
    
    sol = solver.solve(only_one_const=1, order_const=1, tw_penalty_const=1,
                        solver_type='test', num_reads=1, limit_radius=10)

    num_customers = len(problem.customer_ids) # 3
    num_vehicles = len(problem.capacities) # 2
    avg_per_vehicle = math.ceil(num_customers / num_vehicles) # ceil(3/2) = 2
    custom_limit_radius = 10
    expected_k_max = avg_per_vehicle + custom_limit_radius # 2 + 10 = 12

    assert problem.last_args == ([expected_k_max, expected_k_max], 1, 1, 1)



# ------------------- VRPSolution Tests -------------------

def test_vrpsolution_total_cost():
    nodes = {
        'depot': DummyNode(0,0,100),
        'c1': DummyNode(0,0,100),
        'c2': DummyNode(0,0,100)
    }
    costs = {
        'depot': {'c1':5,'c2':10},
        'c1': {'c2':3,'depot':7},
        'c2': {'c1':3,'depot':8}
    }
    time_costs = costs
    capacities = [10]
    customer_ids = ['c1','c2']
    demands = {'c1':1,'c2':1}
    problem = VRPProblem(nodes,'depot',costs,time_costs,capacities,customer_ids,demands)
    sample = {(0,'c1',1):1,(0,'c2',2):1}
    sol = VRPSolution(problem, sample, [2])
    assert sol.total_cost() == 16


def test_vrpsolution_multi_vehicle_cost_and_check():
    # Two vehicles, two customers
    nodes = {
        'depot': DummyNode(0,0,100),
        'c1': DummyNode(0,0,100),
        'c2': DummyNode(0,0,100)
    }
    costs = {
        'depot': {'c1':2,'c2':3},
        'c1': {'depot':2},
        'c2': {'depot':3}
    }
    time_costs = costs
    capacities = [5,5]
    customer_ids = ['c1','c2']
    demands = {'c1':1,'c2':1}
    problem = VRPProblem(nodes,'depot',costs,time_costs,capacities,customer_ids,demands)
    sample = {(0,'c1',1):1,(1,'c2',1):1}
    sol = VRPSolution(problem,sample,[1,1])
    # cost: depot->c1->depot + depot->c2->depot = (2+2)+(3+3)=10
    assert sol.total_cost() == 10
    assert sol.check() is True

# ------------------- QUBO Generation Time-Window Penalty Tests -------------------

def test_get_qubo_time_window_penalties():
    # Setup multi-node TW scenario
    nodes = {
        'depot': DummyNode(0,0,100),
        'j1':    DummyNode(0,0,0),
        'j2':    DummyNode(0,0,0)
    }
    costs = {
        'depot': {'depot':0,'j1':1,'j2':1},
        'j1':    {'depot':1,'j1':0,'j2':10},
        'j2':    {'depot':1,'j1':1,'j2':0}
    }
    time_costs = costs
    problem = VRPProblem(nodes,'depot',costs,time_costs,[10],['j1','j2'],{'j1':1,'j2':1})
    tw_penalty = 5
    qubo = problem.get_qubo([2], only_one_const=0, order_const=0, tw_penalty_const=tw_penalty)

    # j1->j2 transition penalty
    assert qubo.dict.get(((0,'j1',1),(0,'j2',2))) == tw_penalty
    # depot->j2 linear penalty
    assert qubo.dict.get(((0,'j2',1),(0,'j2',1))) == tw_penalty
    # depot->j1 linear penalty
    assert qubo.dict.get(((0,'j1',1),(0,'j1',1))) == tw_penalty
    # j2->j1 violation penalty
    assert qubo.dict.get(((0,'j2',1),(0,'j1',2))) == tw_penalty
    # unrelated key defaults to zero
    assert qubo.dict.get(((99,'x',1),(0,'j1',2)),0) == 0
