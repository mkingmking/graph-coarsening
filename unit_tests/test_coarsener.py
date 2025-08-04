import pytest
import math
from graph import Graph, compute_euclidean_tau
from node import Node
from coarsener import SpatioTemporalGraphCoarsener

@pytest.fixture
def simple_ab_graph():
    # D --(5)--> A --(5)--> B --(5)--> D
    # Nodes: D(0,0), A(5,0), B(10,0)
    # Time windows: D[0,100], A[0,10], B[3,15]
    # Service times: D=0, A=1, B=2
    # Demands: D=0, A=1, B=1
    graph = Graph()
    graph.add_node(Node("D", 0, 0, 0, 0, 100, 0))
    graph.add_node(Node("A", 5, 0, 1, 0, 10, 1))
    graph.add_node(Node("B", 10, 0, 2, 3, 15, 1))

    graph.add_edge("D", "A", compute_euclidean_tau(graph.nodes["D"], graph.nodes["A"]))
    graph.add_edge("A", "B", compute_euclidean_tau(graph.nodes["A"], graph.nodes["B"]))
    graph.add_edge("B", "D", compute_euclidean_tau(graph.nodes["B"], graph.nodes["D"]))
    graph.add_edge("D", "B", compute_euclidean_tau(graph.nodes["D"], graph.nodes["B"])) # Add direct D-B edge
    graph.add_edge("D", "D", 0) # Self-loop for depot
    graph.add_edge("A", "D", compute_euclidean_tau(graph.nodes["A"], graph.nodes["D"])) # A-D edge
    graph.add_edge("B", "A", compute_euclidean_tau(graph.nodes["B"], graph.nodes["A"])) # B-A edge

    return graph

def test_evaluate_feasibility_ab(simple_ab_graph):
    coarsener = SpatioTemporalGraphCoarsener(simple_ab_graph, alpha=1, beta=1, P=0.5, radiusCoeff=1, depot_id="D")
    A = simple_ab_graph.nodes["A"]
    B = simple_ab_graph.nodes["B"]
    tau_ab = compute_euclidean_tau(A, B) # Should be 5

    # feas A->B: (A.e + A.s + tau_ab) <= B.l
    # (0 + 1 + 5) <= 15  => 6 <= 15 => True
    feas_a_b, feas_b_a = coarsener._evaluate_feasibility(simple_ab_graph, A, B)
    assert feas_a_b is True

    # feas B->A: (B.e + B.s + tau_ba) <= A.l
    # (3 + 2 + 5) <= 10  => 10 <= 10 => True
    # (Previous test expectation was False, but with corrected formula, it's True)
    assert feas_b_a is True

def test_compute_slacks_and_order_ab(simple_ab_graph):
    coarsener = SpatioTemporalGraphCoarsener(simple_ab_graph, alpha=1, beta=1, P=0.5, radiusCoeff=1, depot_id="D")
    A = simple_ab_graph.nodes["A"]
    B = simple_ab_graph.nodes["B"]
    tau_ab = compute_euclidean_tau(A, B) # Should be 5

    # slack_A->B = B.l - (A.e + A.s + tau_ab)
    #            = 15 - (0 + 1 + 5) = 15 - 6 = 9
    # slack_B->A = A.l - (B.e + B.s + tau_ba)
    #            = 10 - (3 + 2 + 5) = 10 - 10 = 0
    order, slack = coarsener._compute_slacks_and_order(simple_ab_graph, A, B)
    assert order == "A -> B"
    assert pytest.approx(slack) == 9.0 # Corrected expected slack

def test_compute_new_window_ab(simple_ab_graph):
    coarsener = SpatioTemporalGraphCoarsener(simple_ab_graph, alpha=1, beta=1, P=0.5, radiusCoeff=1, depot_id="D")
    A = simple_ab_graph.nodes["A"]
    B = simple_ab_graph.nodes["B"]
    pi_order = "A -> B"
    tau_ab = compute_euclidean_tau(A, B) # Should be 5

    # e'_SN = max(e_A, e_B - (s_A + tau_AB))
    #       = max(0, 3 - (1 + 5)) = max(0, 3 - 6) = max(0, -3) = 0
    # l'_SN = min(l_A, l_B - (s_A + tau_AB))
    #       = min(10, 15 - (1 + 5)) = min(10, 15 - 6) = min(10, 9) = 9
    e_prime, l_prime = coarsener._compute_new_window(simple_ab_graph, A, B, pi_order)
    assert pytest.approx(e_prime) == 0.0
    assert pytest.approx(l_prime) == 9.0 # Corrected expected l_prime

