from pathlib import Path

from graph_coarsening.coarsener import SpatioTemporalGraphCoarsener
from graph_coarsening.graph import Graph, compute_euclidean_tau
from graph_coarsening.node import Node
from graph_coarsening.quantum_solvers.vrp_problem import VRPProblem
from graph_coarsening.quantum_solvers.vrp_solution import VRPSolution
from graph_coarsening.utils import calculate_route_metrics, load_graph_from_csv


def _build_small_graph() -> Graph:
    graph = Graph()
    graph.add_node(Node("D", 0, 0, 0, 0, 100, 0))
    graph.add_node(Node("A", 3, 0, 2, 0, 20, 1))
    graph.add_node(Node("B", 6, 0, 2, 0, 25, 1))

    for left in graph.nodes.values():
        for right in graph.nodes.values():
            if left.id >= right.id:
                continue
            graph.add_edge(left.id, right.id, compute_euclidean_tau(left, right))

    return graph


def test_graph_add_edge_deduplicates_undirected_pairs():
    graph = Graph()
    graph.add_node(Node("A", 0, 0, 0, 0, 10, 0))
    graph.add_node(Node("B", 1, 0, 0, 0, 10, 0))

    graph.add_edge("A", "B", 1.0)
    graph.add_edge("B", "A", 1.0)

    assert len(graph.edges) == 1
    assert graph.get_neighbors("A") == {"B"}
    assert graph.get_neighbors("B") == {"A"}


def test_graph_remove_node_cleans_edges_and_adjacency():
    graph = _build_small_graph()

    graph.remove_node("A")

    assert "A" not in graph.nodes
    assert "A" not in graph.adj
    assert all(edge.u_id != "A" and edge.v_id != "A" for edge in graph.edges)
    assert "A" not in graph.get_neighbors("D")
    assert "A" not in graph.get_neighbors("B")


def test_coarsener_inflate_route_replays_merge_order():
    graph = _build_small_graph()
    coarsener = SpatioTemporalGraphCoarsener(
        graph, alpha=1.0, beta=1.0, P=0.5, radiusCoeff=1.0, depot_id="D"
    )
    coarsener.merge_layers = [("SN_A_B", "A", "B", "A -> B")]

    inflated = coarsener.inflate_route([["D", "SN_A_B", "D"]])

    assert inflated == [["D", "A", "B", "D"]]


def test_calculate_route_metrics_marks_open_route_infeasible():
    graph = _build_small_graph()

    metrics = calculate_route_metrics(graph, [["D", "A"]], depot_id="D", vehicle_capacity=10)

    assert metrics["is_feasible"] is False
    assert metrics["num_vehicles"] == 1
    assert metrics["total_distance"] == 3.0


def test_load_graph_from_csv_uses_family_capacity_and_builds_complete_graph(tmp_path):
    csv_path = tmp_path / "C101.csv"
    csv_path.write_text(
        "\n".join(
            [
                "NAME,TEST",
                "COMMENT,TEST",
                "TYPE,VRPTW",
                "CUST NO.,XCOORD.,YCOORD.,DEMAND,READY TIME,DUE DATE,SERVICE TIME",
                "0,0,0,0,0,100,0",
                "1,10,0,5,0,100,2",
                "2,20,0,4,0,100,3",
            ]
        )
    )

    graph, depot_id, capacity = load_graph_from_csv(str(csv_path))

    assert depot_id == "0"
    assert capacity == 200.0
    assert set(graph.nodes) == {"0", "1", "2"}
    assert len(graph.edges) == 3


def test_vrp_solution_check_and_total_cost_with_explicit_solution():
    problem = VRPProblem(
        source_depot=0,
        costs=[
            [0, 3, 6],
            [3, 0, 3],
            [6, 3, 0],
        ],
        time_costs=[
            [0, 3, 6],
            [3, 0, 3],
            [6, 3, 0],
        ],
        capacities=[5],
        dests=[1, 2],
        weights={1: 2, 2: 2},
        time_windows={0: (0, 50), 1: (0, 20), 2: (0, 30)},
        service_times={0: 0, 1: 1, 2: 1},
    )

    solution = VRPSolution(problem, sample={}, vehicle_k_limits=[2], solution=[[1, 2]])

    assert solution.check() is True
    assert solution.total_cost() == 12


def test_node_central_time_matches_constructor_formula():
    node = Node("A", 0, 0, 4, 10, 30, 1)

    assert node.t == 18


def test_node_central_time_falls_back_to_e_when_window_too_tight():
    # l - s < 0 branch: service time exceeds latest deadline
    node = Node("B", 0, 0, 15, 5, 10, 1)  # l - s = 10 - 15 = -5 < 0

    assert node.t == node.e


_METRICS_KEYS = {
    "total_distance",
    "total_service_time",
    "total_waiting_time",
    "total_route_duration",
    "time_window_violations",
    "capacity_violations",
    "is_feasible",
    "num_vehicles",
    "total_demand_served",
    "routes_list",
}


def test_calculate_route_metrics_returns_all_keys_on_valid_route():
    graph = _build_small_graph()
    metrics = calculate_route_metrics(graph, [["D", "A", "B", "D"]], depot_id="D", vehicle_capacity=10)

    assert set(metrics.keys()) == _METRICS_KEYS


def test_calculate_route_metrics_returns_all_keys_on_empty_routes():
    graph = _build_small_graph()
    metrics = calculate_route_metrics(graph, [], depot_id="D", vehicle_capacity=10)

    assert set(metrics.keys()) == _METRICS_KEYS
    assert metrics["num_vehicles"] == 0
    assert metrics["is_feasible"] is False
