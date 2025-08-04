import pytest
from graph import Graph, compute_euclidean_tau
from node import Node
from greedy_solver import GreedySolver
from savings_solver import SavingsSolver
from utils import calculate_route_metrics # Ensure this is imported for metrics calculation

@pytest.fixture
def simple_chain_graph():
    # D --(1)--> A --(1)--> B --(2)--> D
    # Nodes: D(0,0), A(1,0), B(2,0)
    # Time windows: D[0,100], A[0,10], B[0,20]
    # Service times: D=0, A=1, B=1
    # Demands: D=0, A=1, B=1
    graph = Graph()
    graph.add_node(Node("D", 0, 0, 0, 0, 100, 0))
    graph.add_node(Node("A", 1, 0, 1, 0, 10, 1))
    graph.add_node(Node("B", 2, 0, 1, 0, 20, 1))

    # Add all necessary edges for a complete graph for simplicity in tests
    node_ids = list(graph.nodes.keys())
    for i in range(len(node_ids)):
        for j in range(i + 1, len(node_ids)):
            id1 = node_ids[i]
            id2 = node_ids[j]
            node1 = graph.nodes[id1]
            node2 = graph.nodes[id2]
            tau = compute_euclidean_tau(node1, node2)
            graph.add_edge(id1, id2, tau)
    return graph

@pytest.fixture
def two_customer_graph():
    # D --(1)--> A --(1)--> B --(1)--> D
    # Nodes: D(0,0), A(1,0), B(2,0)
    # Time windows: D[0,100], A[0,10], B[0,20]
    # Service times: D=0, A=1, B=1
    # Demands: D=0, A=1, B=1
    graph = Graph()
    graph.add_node(Node("D", 0, 0, 0, 0, 100, 0))
    graph.add_node(Node("A", 1, 0, 1, 0, 10, 1))
    graph.add_node(Node("B", 2, 0, 1, 0, 20, 1))

    node_ids = list(graph.nodes.keys())
    for i in range(len(node_ids)):
        for j in range(i + 1, len(node_ids)):
            id1 = node_ids[i]
            id2 = node_ids[j]
            node1 = graph.nodes[id1]
            node2 = graph.nodes[id2]
            tau = compute_euclidean_tau(node1, node2)
            graph.add_edge(id1, id2, tau)
    return graph


# --- Greedy Solver Tests ---

def test_greedy_single_vehicle(simple_chain_graph):
    solver = GreedySolver(simple_chain_graph, depot_id="D", vehicle_capacity=10)
    routes, metrics = solver.solve()
    # Expect one route: D -> A -> B -> D
    assert routes == [["D", "A", "B", "D"]]
    # Distance: D-A (1) + A-B (1) + B-D (2) = 4
    assert pytest.approx(metrics["total_distance"], rel=1e-6) == 4.0
    assert metrics["num_vehicles"] == 1
    assert metrics["is_feasible"] is True

def test_greedy_multiple_vehicles_due_capacity(simple_chain_graph):
    # capacity 1: can only serve one customer per vehicle
    solver = GreedySolver(simple_chain_graph, depot_id="D", vehicle_capacity=1)
    routes, metrics = solver.solve()
    # It will visit A first (closest), return, then B
    # Order of routes might vary, so check content
    expected_routes = [["D", "A", "D"], ["D", "B", "D"]]
    assert {tuple(r) for r in routes} == {tuple(r) for r in expected_routes}
    assert metrics["num_vehicles"] == 2
    assert metrics["is_feasible"] is True

def test_greedy_stuck_when_no_feasible_next(simple_chain_graph):
    # Make B unreachable by time window: e_B = 100, l_B=101, but travel+service makes it too late
    g = simple_chain_graph
    g.nodes["B"].e = 100
    g.nodes["B"].l = 101 # Very tight window, making it hard to reach
    # Depot L is 100. If arrival at depot from A+B is > 100, it's infeasible.
    # D(0,0) A(1,0) B(2,0)
    # D.e=0, D.s=0, D.l=100
    # A.e=0, A.s=1, A.l=10
    # B.e=100, B.s=1, B.l=101

    # Route D->A:
    # Arrive A: 0 + 1 = 1. Service start A: max(1, 0) = 1. Finish A: 1 + 1 = 2.
    # From A, try to go to B:
    # Travel A->B: 1. Arrival B: 2 + 1 = 3. Service start B: max(3, 100) = 100. Finish B: 100 + 1 = 101.
    # From B, try to go to D:
    # Travel B->D: 2. Arrival D: 101 + 2 = 103.
    # 103 > D.l (100) -> Infeasible to return from B.
    # So, A should be visited, then vehicle returns to D, B is unvisited.

    solver = GreedySolver(g, depot_id="D", vehicle_capacity=10)
    routes, metrics = solver.solve()
    
    # First vehicle visits A then cannot feasibly reach B and return to D, so it returns from A.
    # Second vehicle: tries B but can't (no feasible path to B and return), so it gets stuck.
    # The solver should then break the outer loop, leaving B unvisited.
    
    assert len(routes) == 1
    assert routes[0] == ["D", "A", "D"] # Only A is visited
    assert metrics["is_feasible"] is True # Route D-A-D is feasible (no TW/capacity violations on this route)
    assert metrics["total_demand_served"] == g.nodes["A"].demand
    # Note: total_demand_served will not equal total problem demand, but the generated route is feasible.

def test_greedy_no_customers():
    graph = Graph()
    graph.add_node(Node("D", 0, 0, 0, 0, 100, 0))
    solver = GreedySolver(graph, depot_id="D", vehicle_capacity=10)
    routes, metrics = solver.solve()
    assert routes == []
    assert metrics["num_vehicles"] == 0
    assert metrics["total_distance"] == 0.0
    assert metrics["is_feasible"] is True # No customers, no routes, thus no violations.

# --- Savings Solver Tests ---

def test_savings_solver_merges(two_customer_graph):
    solver = SavingsSolver(two_customer_graph, depot_id="D", vehicle_capacity=10)
    routes, metrics = solver.solve()
    # Should merge into one route: D -> A -> B -> D
    assert len(routes) == 1
    assert routes[0] == ["D", "A", "B", "D"]
    assert metrics["num_vehicles"] == 1
    assert metrics["is_feasible"] is True
    # Distance: D-A (1) + A-B (1) + B-D (2) = 4
    assert pytest.approx(metrics["total_distance"], rel=1e-6) == 4.0

def test_savings_solver_no_merge_when_infeasible(two_customer_graph):
    # Tight time window on B so merge A-B is infeasible
    g = two_customer_graph
    g.nodes["B"].e = 100
    g.nodes["B"].l = 101 # Very tight window for B
    g.nodes["D"].l = 100 # Ensure depot time window is tight enough to cause issue

    # Test merge A->B:
    # D-A-B-D
    # Arrive A: 1. Finish A: 1+1=2.
    # Travel A->B: 1. Arrive B: 2+1=3. Service Start B: max(3, 100) = 100. Finish B: 100+1=101.
    # Travel B->D: 2. Arrive D: 101+2=103.
    # 103 > D.l (100) -> Infeasible. So A-B merge should not happen.

    solver = SavingsSolver(g, depot_id="D", vehicle_capacity=10)
    routes, metrics = solver.solve()
    
    # Should remain two separate routes
    expected = [["D", "A", "D"], ["D", "B", "D"]]
    # Order of routes list may vary, so compare as sets of tuples
    assert {tuple(r) for r in routes} == {tuple(r) for r in expected}
    assert metrics["num_vehicles"] == 2
    # Route D-A-D is feasible.
    # Route D-B-D: Arrive B at 2, service starts at 100 (due to B.e). Finish B at 101. Travel B->D (2). Arrive D at 103.
    # Since D.l is 100, 103 > 100, so route D-B-D is INFEASIBLE.
    # Therefore, overall metrics["is_feasible"] should be False.
    assert metrics["is_feasible"] is False 
    # Total distance: (D-A-D) + (D-B-D) = (1+1) + (2+2) = 2 + 4 = 6
    assert pytest.approx(metrics["total_distance"], rel=1e-6) == 6.0

def test_savings_solver_no_customers():
    graph = Graph()
    graph.add_node(Node("D", 0, 0, 0, 0, 100, 0))
    solver = SavingsSolver(graph, depot_id="D", vehicle_capacity=10)
    routes, metrics = solver.solve()
    assert routes == []
    assert metrics["num_vehicles"] == 0
    assert metrics["total_distance"] == 0.0
    assert metrics["is_feasible"] is True # No customers, no routes, thus no violations.

