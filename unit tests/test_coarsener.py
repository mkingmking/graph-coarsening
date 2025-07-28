import pytest
from node import Node
from edge import Edge
from graph import Graph
from utils import compute_euclidean_tau
from coarsener import SpatioTemporalGraphCoarsener
from savings_solver import SavingsSolver
from greedy_solver import GreedySolver
@pytest.fixture
def simple_ab_graph():
    # Nodes A and B with known windows and demands
    g = Graph()
    g.add_node(Node("D", 0, 0, s=0, e=0, l=100, demand=0))   # depot
    g.add_node(Node("A", 0, 0, s=1, e=0, l=10, demand=1))
    g.add_node(Node("B", 3, 4, s=2, e=3, l=15, demand=2))
    # fully connect
    g.add_edge("D", "A", compute_euclidean_tau(g.nodes["D"], g.nodes["A"]))
    g.add_edge("D", "B", compute_euclidean_tau(g.nodes["D"], g.nodes["B"]))
    g.add_edge("A", "B", compute_euclidean_tau(g.nodes["A"], g.nodes["B"]))
    return g

def test_compute_D_ij_ab(simple_ab_graph):
    coarsener = SpatioTemporalGraphCoarsener(simple_ab_graph, alpha=1, beta=2, P=0.5, radiusCoeff=1, depot_id="D")
    edge_ab = simple_ab_graph.get_edge_by_nodes("A", "B")
    D_ab = coarsener._compute_D_ij(simple_ab_graph, edge_ab)
    # tau = 5, temporal_slack = max(0, e_B - (t_A + s_A + tau))
    # t_A = (0 + (10-1))/2 = 4.5, s_A=1 => e_B - (4.5+1+5)=3 -10.5 = -7.5 =>0
    assert pytest.approx(D_ab) == 5.0

def test_evaluate_feasibility_ab(simple_ab_graph):
    coarsener = SpatioTemporalGraphCoarsener(simple_ab_graph, alpha=1, beta=1, P=0.5, radiusCoeff=1, depot_id="D")
    A = simple_ab_graph.nodes["A"]
    B = simple_ab_graph.nodes["B"]
    feas_a_b, feas_b_a = coarsener._evaluate_feasibility(simple_ab_graph, A, B)
    # feas A->B: 0 <= 15 - 2 - 5 - 1 => 0 <= 7  => True
    # feas B->A: 3 <= 10 - 1 - 5 - 2 => 3 <= 2 => False
    assert feas_a_b is True
    assert feas_b_a is False

def test_compute_slacks_and_order_ab(simple_ab_graph):
    coarsener = SpatioTemporalGraphCoarsener(simple_ab_graph, alpha=1, beta=1, P=0.5, radiusCoeff=1, depot_id="D")
    A = simple_ab_graph.nodes["A"]
    B = simple_ab_graph.nodes["B"]
    order, slack = coarsener._compute_slacks_and_order(simple_ab_graph, A, B)
    # slack_A->B = (15-2-5) - (0+1) = 8 -1 =7
    assert order == "A -> B"
    assert pytest.approx(slack) == 7.0

def test_compute_new_window_ab(simple_ab_graph):
    coarsener = SpatioTemporalGraphCoarsener(simple_ab_graph, alpha=1, beta=1, P=0.5, radiusCoeff=1, depot_id="D")
    A = simple_ab_graph.nodes["A"]
    B = simple_ab_graph.nodes["B"]
    pi_order = "A -> B"
    e_prime, l_prime = coarsener._compute_new_window(simple_ab_graph, A, B, pi_order)
    # e' = max(e_A, e_B - (s_A + tau)) = max(0, 3-(1+5)) = 0
    # l' = min(l_A + s_B + tau, l_B) = min(10+2+5,15) = 15
    assert pytest.approx(e_prime) == 0.0
    assert pytest.approx(l_prime) == 15.0

def test_reconnect_neighbors_conservatively():
    # Build a little graph where A and B both connect to C
    g = Graph()
    g.add_node(Node("D",0,0,0,0,100,0))
    g.add_node(Node("A",0,0,1,0,10,1))
    g.add_node(Node("B",3,4,2,3,15,2))
    g.add_node(Node("C",6,8,1,0,20,3))
    # Edges: A-C, B-C and connect to depot
    for u,v in [("A","C"),("B","C"),("A","D"),("B","D")]:
        g.add_edge(u, v, compute_euclidean_tau(g.nodes[u], g.nodes[v]))
    coarsener = SpatioTemporalGraphCoarsener(g, alpha=1, beta=1, P=0.5, radiusCoeff=1, depot_id="D")
    # create a super-node manually
    A = g.nodes["A"]
    B = g.nodes["B"]
    mid_x, mid_y = (A.x+B.x)/2, (A.y+B.y)/2
    super_node = Node("SN", mid_x, mid_y, A.s+B.s, 0, 0, A.demand+B.demand, is_super_node=True)
    g.add_node(super_node)
    # Before reconnect: no edge between SN and C
    assert g.get_edge_by_nodes("SN", "C") is None
    coarsener._reconnect_neighbors_conservatively(g, super_node, A, B)
    # After reconnect, SN-C should exist
    edge = g.get_edge_by_nodes("SN", "C")
    assert isinstance(edge, Edge)
    # check tau matches Euclidean distance
    assert pytest.approx(edge.tau) == compute_euclidean_tau(super_node, g.nodes["C"])

def test_coarsen_and_inflate_full_cycle(simple_ab_graph):
    # set P high so exactly one merge
    co = SpatioTemporalGraphCoarsener(simple_ab_graph, alpha=1, beta=1, P=0.9, radiusCoeff=10, depot_id="D")
    G_coarse, layers = co.coarsen()
    # should have merged A and B into SN_A_B
    assert any(sn.startswith("SN_A_B") for sn, *_ in layers)
    # coarse graph now has D and SN_A_B
    assert set(G_coarse.nodes) == {"D", "SN_A_B"}
    # test inflate_route: one route on coarse graph
    coarse_route = [["D", "SN_A_B", "D"]]
    inflated = co.inflate_route(coarse_route)
    # should replace SN_A_B by A,B in that order
    assert inflated == [["D", "A", "B", "D"]]
