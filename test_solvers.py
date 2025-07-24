import pytest
from graph_coarsening import (
    Node,
    Graph,
    compute_euclidean_tau,
    GreedySolver,
    SavingsSolver,
    calculate_route_metrics,
)

@pytest.fixture
def simple_chain_graph():
    """
    Graph: D—A—B in a line.
    Depot D at (0,0), A at (1,0), B at (2,0).
    Service times zero, time windows [0, 100], demands 1 each.
    """
    g = Graph()
    g.add_node(Node("D", 0, 0, s=0, e=0, l=100, demand=0))
    g.add_node(Node("A", 1, 0, s=0, e=0, l=100, demand=1))
    g.add_node(Node("B", 2, 0, s=0, e=0, l=100, demand=1))
    # Fully connect
    for u, v in [("D", "A"), ("A", "B"), ("D", "B")]:
        tau = compute_euclidean_tau(g.nodes[u], g.nodes[v])
        g.add_edge(u, v, tau)
    return g

def test_greedy_single_vehicle(simple_chain_graph):
    solver = GreedySolver(simple_chain_graph, depot_id="D", vehicle_capacity=10)
    routes, metrics = solver.solve()
    # Expect one route: D -> A -> B -> D
    assert routes == [["D", "A", "B", "D"]]
    # Distance = 1 + 1 + 2 = 4
    assert pytest.approx(metrics["total_distance"], rel=1e-6) == 4.0
    assert metrics["num_vehicles"] == 1
    assert metrics["is_feasible"] is True
    assert metrics["total_demand_served"] == 2.0

def test_greedy_multiple_vehicles_due_capacity(simple_chain_graph):
    # capacity 1: can only serve one customer per vehicle
    solver = GreedySolver(simple_chain_graph, depot_id="D", vehicle_capacity=1)
    routes, metrics = solver.solve()
    # It will visit A first (closest), return, then B
    assert routes == [["D", "A", "D"], ["D", "B", "D"]]
    assert metrics["num_vehicles"] == 2
    assert pytest.approx(metrics["total_distance"], rel=1e-6) == (
        compute_euclidean_tau(simple_chain_graph.nodes["D"], simple_chain_graph.nodes["A"]) * 2
        + compute_euclidean_tau(simple_chain_graph.nodes["D"], simple_chain_graph.nodes["B"]) * 2
    )
    assert metrics["total_demand_served"] == 2.0
    assert metrics["is_feasible"] is True

def test_greedy_stuck_when_no_feasible_next(simple_chain_graph):
    # Make B unreachable by time window: e_B = 50, l_B=60, but travel+service makes it too late
    g = simple_chain_graph
    g.nodes["B"].e = 100
    g.nodes["B"].l = 101
    solver = GreedySolver(g, depot_id="D", vehicle_capacity=10)
    routes, metrics = solver.solve()
    # First vehicle visits A then cannot reach B, returns
    # Second vehicle: tries B but can't (no feasible), so stuck -> breaks
    assert routes[0][1:] == ["A", "D"]
    # Second route will be just ["D"] (depot only, no visits)
    assert routes[1] == ["D"]
    # Only A served
    assert metrics["total_demand_served"] == 1.0
    assert metrics["is_feasible"] is False  # because B never served

# --- SavingsSolver tests ---

@pytest.fixture
def two_customer_graph():
    """
    Graph: D, A, B in a line at (0,0), (3,0), (5,0).
    Time windows wide, demands 1 each.
    """
    g = Graph()
    g.add_node(Node("D", 0, 0, s=0, e=0, l=100, demand=0))
    g.add_node(Node("A", 3, 0, s=0, e=0, l=100, demand=1))
    g.add_node(Node("B", 5, 0, s=0, e=0, l=100, demand=1))
    # Fully connect edges
    for u, v in [("D", "A"), ("A", "B"), ("D", "B")]:
        tau = compute_euclidean_tau(g.nodes[u], g.nodes[v])
        g.add_edge(u, v, tau)
    return g

def test_calculate_savings(two_customer_graph):
    solver = SavingsSolver(two_customer_graph, depot_id="D", vehicle_capacity=10)
    savings = solver._calculate_savings()
    # Only one pair (A,B)
    assert len(savings) == 1
    s_value, i, j = savings[0]
    # s = d(D,A) + d(B,D) - d(A,B) = 3 + 5 - 2 = 6
    assert i == "A" and j == "B"
    assert pytest.approx(s_value, rel=1e-6) == 6.0

def test_check_merge_feasibility(two_customer_graph):
    solver = SavingsSolver(two_customer_graph, depot_id="D", vehicle_capacity=2)
    # routes before merge
    rA = ["D", "A", "D"]
    rB = ["D", "B", "D"]
    assert solver._check_merge_feasibility(rA, rB, "A", "B") is True
    # If capacity too small
    solver_small = SavingsSolver(two_customer_graph, depot_id="D", vehicle_capacity=1)
    assert solver_small._check_merge_feasibility(rA, rB, "A", "B") is False

def test_savings_solver_merges(two_customer_graph):
    solver = SavingsSolver(two_customer_graph, depot_id="D", vehicle_capacity=10)
    routes, metrics = solver.solve()
    # Should merge into one route: D -> A -> B -> D
    assert routes == [["D", "A", "B", "D"]]
    assert metrics["num_vehicles"] == 1
    assert metrics["is_feasible"] is True
    # Distance = 3 + 2 + 5 = 10
    assert pytest.approx(metrics["total_distance"], rel=1e-6) == 10.0

def test_savings_solver_no_merge_when_infeasible(two_customer_graph):
    # Tight time window on B so merge infeasible
    g = two_customer_graph
    g.nodes["B"].e = 100
    g.nodes["B"].l = 101
    solver = SavingsSolver(g, depot_id="D", vehicle_capacity=10)
    routes, metrics = solver.solve()
    # Should remain two separate routes
    expected = [["D", "A", "D"], ["D", "B", "D"]]
    # Order of routes list may vary, so compare as sets of tuples
    assert {tuple(r) for r in routes} == {tuple(r) for r in expected}
    assert metrics["num_vehicles"] == 2
    assert metrics["is_feasible"] is False  # B route violates its window in aggregate metrics
