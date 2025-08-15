import pytest

from quantum_solvers.vrp_solvers import FullQuboSolver, AveragePartitionSolver, DWaveSolvers_modified
from quantum_solvers.vrp_solution import VRPSolution
from quantum_solvers.vrp_problem_qubo import VRPProblem
from quantum_solvers.qubo_solver import Qubo

# Helper classes for testing
class FakeProblem:
    def __init__(self, customer_ids, capacities):
        self.customer_ids = customer_ids
        self.capacities = capacities
        self.get_qubo_called = False
        self.last_args = None

    def get_qubo(self, vehicle_k_limits, only_one_const, order_const, tw_penalty_const):
        # Record that get_qubo was called with these parameters
        self.get_qubo_called = True
        self.last_args = (vehicle_k_limits, only_one_const, order_const, tw_penalty_const)
        # Return a dummy QUBO object
        return 'dummy_qubo'


def test_full_qubo_solver_calls_get_qubo_and_returns_solution(monkeypatch):
    # Setup a fake problem with one customer and one vehicle
    problem = FakeProblem(['c1'], [10])
    solver = FullQuboSolver(problem)

    # Monkey-patch the QUBO solver to return a known sample
    def fake_solve_qubo(qubo, solver_type, limit, num_reads):
        # Verify that the QUBO instance and solver parameters are passed correctly
        assert qubo == 'dummy_qubo'
        assert solver_type == 'simulated'
        assert limit == 1
        assert num_reads == 5
        # Return a sample indicating the vehicle visits the customer at step 1
        return [ {(0, 'c1', 1): 1} ]

    monkeypatch.setattr(DWaveSolvers_modified, 'solve_qubo', fake_solve_qubo)

    # Call solve and verify behavior
    sol = solver.solve(only_one_const=0.1, order_const=0.2, tw_penalty_const=0.3,
                       solver_type='simulated', num_reads=5)

    # Ensure get_qubo was called with correct arguments
    assert problem.get_qubo_called
    assert problem.last_args == ([1], 0.1, 0.2, 0.3)

    # Verify the returned solution
    assert isinstance(sol, VRPSolution)
    assert sol.vehicle_k_limits == [1]
    assert sol.solution == [['c1']]


def test_full_qubo_solver_empty_samples(monkeypatch):
    # Setup a fake problem with two customers and two vehicles
    problem = FakeProblem(['c1', 'c2'], [10, 10])
    solver = FullQuboSolver(problem)

    # Monkey-patch the QUBO solver to return no samples
    monkeypatch.setattr(DWaveSolvers_modified, 'solve_qubo', lambda qubo, solver_type, limit, num_reads: [])

    # Call solve
    sol = solver.solve(only_one_const=1, order_const=2, tw_penalty_const=3,
                       solver_type='any', num_reads=1)

    # For FQS, k_max should equal number of customers (2)
    assert sol.vehicle_k_limits == [2, 2]
    # Solution routes should be empty for each vehicle
    assert sol.solution == [[], []]


def test_average_partition_solver(monkeypatch):
    # Setup a fake problem with three customers and two vehicles
    problem = FakeProblem(['c1', 'c2', 'c3'], [10, 10])
    solver = AveragePartitionSolver(problem)

    # Monkey-patch the QUBO solver to return a known sample for two vehicles
    def fake_solve_qubo(qubo, solver_type, limit, num_reads):
        # Return a sample indicating each vehicle visits one customer
        return [ {(0, 'c1', 1): 1, (1, 'c2', 2): 1} ]

    monkeypatch.setattr(DWaveSolvers_modified, 'solve_qubo', fake_solve_qubo)

    # Call solve with a custom limit_radius
    sol = solver.solve(only_one_const=1, order_const=2, tw_penalty_const=3,
                       solver_type='local', num_reads=10, limit_radius=2)

    # avg_per_vehicle = ceil(3/2) = 2, so k_max = 2 + 2 = 4
    assert problem.get_qubo_called
    assert problem.last_args == ([4, 4], 1, 2, 3)

    # Verify the returned solution
    assert isinstance(sol, VRPSolution)
    assert sol.vehicle_k_limits == [4, 4]
    assert sol.solution == [['c1'], ['c2']]

# Additional tests for edge cases and metrics

def test_vrpsolution_total_cost():
    # Create a simple problem with two customers and one vehicle
    class DummyNode:
        def __init__(self, e, s, l):
            self.e = e
            self.s = s
            self.l = l

    nodes = {
        'depot': DummyNode(e=0, s=0, l=100),
        'c1': DummyNode(e=0, s=0, l=100),
        'c2': DummyNode(e=0, s=0, l=100)
    }
    costs = {
        'depot': {'c1': 5, 'c2': 10},
        'c1': {'c2': 3, 'depot': 7},
        'c2': {'c1': 3, 'depot': 8}
    }
    time_costs = costs  # ignore time in this test
    capacities = [10]
    customer_ids = ['c1', 'c2']
    demands = {'c1': 1, 'c2': 1}

    problem = VRPProblem(nodes, 'depot', costs, time_costs, capacities, customer_ids, demands)
    # Construct a sample where vehicle visits c1 then c2
    sample = {(0, 'c1', 1): 1, (0, 'c2', 2): 1}
    sol = VRPSolution(problem, sample, [2])
    # total cost: depot->c1 (5) + c1->c2 (3) + c2->depot (8) = 16
    assert sol.total_cost() == 16


def test_vrpsolution_check_valid():
    # Reuse the simple problem from total_cost test
    class DummyNode:
        def __init__(self, e, s, l):
            self.e = e
            self.s = s
            self.l = l

    nodes = {
        'depot': DummyNode(e=0, s=0, l=100),
        'c1': DummyNode(e=0, s=0, l=100)
    }
    costs = {'depot': {'c1': 1}, 'c1': {'depot': 1}}
    time_costs = costs
    capacities = [5]
    customer_ids = ['c1']
    demands = {'c1': 1}

    problem = VRPProblem(nodes, 'depot', costs, time_costs, capacities, customer_ids, demands)
    sample = {(0, 'c1', 1): 1}
    sol = VRPSolution(problem, sample, [1])
    assert sol.check() is True


def test_vrpsolution_check_capacity_violation():
    # Demand exceeds capacity
    class DummyNode:
        def __init__(self, e, s, l):
            self.e = e
            self.s = s
            self.l = l

    nodes = {'depot': DummyNode(0,0,100), 'c1': DummyNode(0,0,100)}
    costs = {'depot':{'c1':1}, 'c1':{'depot':1}}
    time_costs = costs
    capacities = [0]
    customer_ids = ['c1']
    demands = {'c1': 5}

    problem = VRPProblem(nodes, 'depot', costs, time_costs, capacities, customer_ids, demands)
    sample = {(0, 'c1', 1): 1}
    sol = VRPSolution(problem, sample, [1])
    assert sol.check() is False


def test_vrpsolution_check_duplicate_visit():
    # Duplicate customer in sample
    class DummyNode:
        def __init__(self, e, s, l):
            self.e = e
            self.s = s
            self.l = l

    nodes = {'depot': DummyNode(0,0,100), 'c1': DummyNode(0,0,100)}
    costs = {'depot':{'c1':1}, 'c1':{'depot':1}}
    time_costs = costs
    capacities = [10]
    customer_ids = ['c1']
    demands = {'c1': 1}

    problem = VRPProblem(nodes, 'depot', costs, time_costs, capacities, customer_ids, demands)
    # Mark same variable twice
    sample = {(0, 'c1', 1): 1, (0, 'c1', 1): 1}
    sol = VRPSolution(problem, sample, [1])
    # VRPSolution.check uses set, duplicates ignored, but missing depot count equal => valid
    assert sol.check() is True


def test_vrpsolution_check_time_window_violation():
    # Time window violation on service
    class DummyNode:
        def __init__(self, e, s, l):
            self.e = e
            self.s = s
            self.l = l

    # j1 service time + travel delays j2 beyond its latest
    nodes = {
        'depot': DummyNode(e=0, s=0, l=100),
        'c1': DummyNode(e=0, s=100, l=100),  # long service
        'c2': DummyNode(e=0, s=0, l=50)      # tight latest time
    }
    costs = {'depot':{'c1':1,'c2':10}, 'c1':{'c2':1}, 'c2':{'depot':1}}
    time_costs = costs
    capacities = [10]
    customer_ids = ['c1','c2']
    demands = {'c1': 1, 'c2': 1}

    problem = VRPProblem(nodes, 'depot', costs, time_costs, capacities, customer_ids, demands)
    sample = {(0, 'c1', 1): 1, (0, 'c2', 2): 1}
    sol = VRPSolution(problem, sample, [2])
    assert sol.check() is False


def test_get_qubo_time_window_penalties():
    # Test QUBO generation for time-window violation penalties
    class DummyNode:
        def __init__(self, e, s, l):
            self.e = e
            self.s = s
            self.l = l

    # Define nodes with specific time windows for testing
    nodes = {
        'depot': DummyNode(e=0, s=0, l=100), # Depot time window
        'j1': DummyNode(e=0, s=0, l=0),     # j1 has a tight time window
        'j2': DummyNode(e=0, s=0, l=0)      # j2 has a tight time window
    }

    # Define a complete cost/time_cost matrix for all node pairs
    # This ensures all lookups in get_qubo are valid.
    # The critical path for violation is j1 -> j2 with time_cost 10.
    # Also, depot -> j2 with time_cost 1 will violate j2's time window.
    costs = {
        'depot': {'depot': 0, 'j1': 1, 'j2': 1},
        'j1':    {'depot': 1, 'j1': 0, 'j2': 10}, # j1->j2 has a high travel time
        'j2':    {'depot': 1, 'j1': 1, 'j2': 0}
    }
    time_costs = costs # For this test, time_costs are the same as costs

    capacities = [10]
    customer_ids = ['j1','j2']
    demands = {'j1': 1, 'j2': 1}

    problem = VRPProblem(nodes, 'depot', costs, time_costs, capacities, customer_ids, demands)
    tw_penalty = 5
    qubo = problem.get_qubo([2], only_one_const=0, order_const=0, tw_penalty_const=tw_penalty)

    # Expected penalties:

    # 1. Penalty for j1 -> j2 transition (customer-to-customer)
    #    Node j1: e=0, s=0, l=0
    #    Node j2: e=0, s=0, l=0
    #    tau_j1_j2 = 10
    #    earliest_arrival_at_j2_from_j1 = j1.e + j1.s + tau_j1_j2 = 0 + 0 + 10 = 10
    #    Since 10 > j2.l (0), this transition should incur a penalty.
    assert qubo.dict[((0, 'j1', 1), (0, 'j2', 2))] == tw_penalty

    # 2. Penalty for depot -> j2 transition (depot-to-customer)
    #    Depot: e=0, s=0, l=100
    #    Node j2: e=0, s=0, l=0
    #    tau_depot_j2 = 1
    #    arrival_at_j2_from_depot = depot.e + tau_depot_j2 = 0 + 1 = 1
    #    Since 1 > j2.l (0), this should incur a linear penalty on x_{0,j2,1}.
    diag_key_j2 = ((0, 'j2', 1), (0, 'j2', 1))
    assert diag_key_j2 in qubo.dict and qubo.dict[diag_key_j2] == tw_penalty

    # 3. Check for depot -> j1 transition (depot-to-customer)
    #    Depot: e=0, s=0, l=100
    #    Node j1: e=0, s=0, l=0
    #    tau_depot_j1 = 1
    #    arrival_at_j1_from_depot = depot.e + tau_depot_j1 = 0 + 1 = 1
    #    Since 1 > j1.l (0), this should also incur a linear penalty on x_{0,j1,1}.
    diag_key_j1 = ((0, 'j1', 1), (0, 'j1', 1))
    assert diag_key_j1 in qubo.dict and qubo.dict[diag_key_j1] == tw_penalty


    # Verify no unexpected penalties for j2 -> j1 (assuming it's feasible or not considered)
    # For j2->j1: j2.e + j2.s + tau_j2_j1 = 0 + 0 + 1 = 1. This is > j1.l (0).
    # So, there should also be a penalty for j2->j1.
    assert qubo.dict[((0, 'j2', 1), (0, 'j1', 2))] == tw_penalty

    # Verify that if only_one_const and order_const are 0, only TW penalties are present
    # Check a field that should only have TW penalty
    assert qubo.dict.get(((0, 'j1', 1), (0, 'j2', 2)), 0) == tw_penalty
    assert qubo.dict.get(diag_key_j2, 0) == tw_penalty
    assert qubo.dict.get(diag_key_j1, 0) == tw_penalty
    assert qubo.dict.get(((0, 'j2', 1), (0, 'j1', 2)), 0) == tw_penalty

    # Check a field that should NOT have any penalty (e.g., if it's not a TW violation)
    # For example, if we had a valid transition, it would be 0 here.
    # Given the tight TWs, most will be penalized. Let's pick a non-existent var to check default 0.
    assert qubo.dict.get(((99, 'non_existent', 1), (0, 'j1', 2)), 0) == 0

