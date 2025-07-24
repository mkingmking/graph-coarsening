import math
import pytest
from graph_coarsening import Node, Edge, Graph, compute_euclidean_tau, calculate_route_metrics

# --- Node tests ---

def test_node_initialization_and_t_calculation():
    # l - s >= 0 case
    n1 = Node(id="A", x=1.23456, y=2.34567, s=3.0, e=4.0, l=6.0, demand=5.0)
    assert n1.id == "A"
    assert pytest.approx(n1.x, rel=1e-3) == 1.23456
    assert pytest.approx(n1.y, rel=1e-3) == 2.34567
    assert n1.s == 3.0
    assert n1.e == 4.0
    assert n1.l == 6.0
    # t = (e + (l - s)) / 2 = (4 + 3) / 2 = 3.5
    assert pytest.approx(n1.t, rel=1e-3) == 3.5
    assert n1.demand == 5.0
    assert not n1.is_super_node
    assert n1.original_nodes == ["A"]

def test_node_t_when_l_less_than_s():
    # l < s should fall back to t = e
    n2 = Node(id="B", x=0, y=0, s=5.0, e=2.0, l=3.0, demand=0.0)
    assert n2.t == 2.0

def test_node_repr():
    n = Node(id="X", x=1.2, y=3.4, s=0.5, e=1.0, l=2.0, demand=0.0)
    rep = repr(n)
    # Check formatting and presence of key fields
    assert "Node(ID=X" in rep
    assert "Coords=(1.20,3.40)" in rep
    assert "S=0.50" in rep
    assert "TW=[1.00,2.00]" in rep
    assert "T=" in rep
    assert "Demand=0.00" in rep

# --- Edge tests ---

def test_edge_initialization_and_defaults():
    e = Edge("A", "B", tau=5.0)
    assert e.u_id == "A"
    assert e.v_id == "B"
    assert e.tau == 5.0
    assert e.D_ij == 0.0

def test_edge_repr():
    e = Edge("A", "B", tau=5.0)
    rep = repr(e)
    assert rep == "Edge(A-B, Tau=5.00, D_ij=0.00)"

# --- Graph tests ---

@pytest.fixture
def simple_graph():
    g = Graph()
    # Create three nodes: depot, C1, C2
    g.add_node(Node("D", 0, 0, s=0, e=0, l=100, demand=0))
    g.add_node(Node("A", 3, 4, s=2, e=5, l=10, demand=1))
    g.add_node(Node("B", 0, 5, s=1, e=0, l=15, demand=2))
    return g

def test_add_node_and_adj(simple_graph):
    g = simple_graph
    assert "D" in g.nodes and "A" in g.nodes and "B" in g.nodes
    assert g.adj["D"] == set()
    # Adding again should not break adjacency
    g.add_node(Node("D", 0, 0, s=0, e=0, l=0, demand=0))
    assert "D" in g.adj

def test_add_edge_and_duplicates(simple_graph):
    g = simple_graph
    # Missing node raises
    with pytest.raises(ValueError):
        g.add_edge("X", "D", 1.0)
    # Valid edge
    g.add_edge("D", "A", 5.0)
    assert len(g.edges) == 1
    assert "A" in g.adj["D"] and "D" in g.adj["A"]
    # Duplicate does nothing
    g.add_edge("A", "D", 5.0)
    assert len(g.edges) == 1

def test_get_edge_by_nodes(simple_graph):
    g = simple_graph
    g.add_edge("D", "B", 5.0)
    edge = g.get_edge_by_nodes("B", "D")
    assert isinstance(edge, Edge)
    assert edge.u_id in ("D","B") and edge.v_id in ("D","B")
    # Nonexistent returns None
    assert g.get_edge_by_nodes("A", "B") is None

def test_get_neighbors_and_all_edges(simple_graph):
    g = simple_graph
    g.add_edge("D", "A", 5.0)
    g.add_edge("D", "B", 5.0)
    assert g.get_neighbors("D") == {"A", "B"}
    assert g.get_neighbors("X") == set()
    edges_d = g.get_all_edges_for_node("D")
    assert len(edges_d) == 2
    # A has only one edge
    assert len(g.get_all_edges_for_node("A")) == 1

def test_remove_node(simple_graph):
    g = simple_graph
    g.add_edge("D", "A", 5.0)
    g.add_edge("A", "B", 6.0)
    g.remove_node("A")
    # A gone
    assert "A" not in g.nodes
    assert "A" not in g.adj
    # Edges involving A removed
    assert all("A" not in (e.u_id, e.v_id) for e in g.edges)
    # Adj lists for neighbors updated
    assert "A" not in g.adj["D"]
    assert "A" not in g.adj["B"]
    # Removing nonexistent should do nothing
    before = dict(g.nodes)
    g.remove_node("Z")
    assert dict(g.nodes) == before

# --- Helper function tests ---

def test_compute_euclidean_tau():
    n1 = Node("P", 0, 0, 0,0,0,0)
    n2 = Node("Q", 3, 4, 0,0,0,0)
    assert compute_euclidean_tau(n1, n2) == pytest.approx(5.0)

# --- calculate_route_metrics tests ---

def test_empty_routes():
    g = Graph()
    # define depot so graph.nodes[...] does not KeyError
    g.nodes["D"] = Node("D",0,0,0,0,100,0)
    metrics = calculate_route_metrics(g, [], "D", vehicle_capacity=10)
    assert metrics == {
        "total_distance": 0.0,
        "total_service_time": 0.0,
        "total_waiting_time": 0.0,
        "total_route_duration": 0.0,
        "time_window_violations": 0,
        "capacity_violations": 0,
        "is_feasible": False,
        "num_vehicles": 0,
        "total_demand_served": 0.0,
        "routes_list": []
    }

def make_test_graph():
    g = Graph()
    # Depot: no service time, wide window
    g.add_node(Node("D", 0, 0, s=0, e=0, l=100, demand=0))
    # Customer at (3,4)
    g.add_node(Node("C", 3, 4, s=2, e=5, l=10, demand=1))
    return g

def test_single_route_feasible():
    g = make_test_graph()
    route = ["D", "C", "D"]
    metrics = calculate_route_metrics(g, [route], "D", vehicle_capacity=10)
    # distance = 5 + 5 = 10
    assert pytest.approx(metrics["total_distance"], rel=1e-6) == 10.0
    assert metrics["total_service_time"] == 2.0
    assert metrics["total_waiting_time"] == 0.0
    assert metrics["time_window_violations"] == 0
    assert metrics["capacity_violations"] == 0
    assert metrics["is_feasible"] is True
    assert metrics["num_vehicles"] == 1
    assert metrics["total_demand_served"] == 1.0
    # route duration = finish time at depot = 12
    assert pytest.approx(metrics["total_route_duration"], rel=1e-6) == 12.0

def test_capacity_violation_and_time_window_violation():
    g = make_test_graph()
    # Make capacity too small and time window too tight
    g.nodes["C"].demand = 5
    g.nodes["C"].l = 5  # arrival at time=5, service start=5 is fine; use smaller l to force violation
    g.nodes["C"].l = 4  # now service_start_time 5 > 4 => time violation
    route = ["D", "C", "D"]
    metrics = calculate_route_metrics(g, [route], "D", vehicle_capacity=1)
    assert metrics["capacity_violations"] == 1
    assert metrics["time_window_violations"] == 1
    assert metrics["is_feasible"] is False

def test_single_node_route_skipped():
    g = make_test_graph()
    # Route with only depot
    metrics = calculate_route_metrics(g, [["D"]], "D", vehicle_capacity=10)
    # Nothing happens but num_vehicles=1, and is_feasible remains True
    assert metrics["total_distance"] == 0.0
    assert metrics["total_service_time"] == 0.0
    assert metrics["is_feasible"] is True
    assert metrics["num_vehicles"] == 1
    assert metrics["routes_list"] == [["D"]]
