import pytest
from graph import Graph, compute_euclidean_tau
from node import Node
from utils import calculate_route_metrics

@pytest.fixture
def sample_graph():
    graph = Graph()
    graph.add_node(Node("D", 0, 0, 0, 0, 100, 0))  # Depot
    graph.add_node(Node("C1", 10, 0, 5, 10, 20, 10)) # Customer 1
    graph.add_node(Node("C2", 20, 0, 3, 15, 25, 5))  # Customer 2
    graph.add_node(Node("C3", 30, 0, 2, 0, 100, 20)) # Customer 3 (flexible TW)

    # Add edges (Euclidean distances)
    nodes_list = list(graph.nodes.values())
    for i in range(len(nodes_list)):
        for j in range(i + 1, len(nodes_list)):
            n1 = nodes_list[i]
            n2 = nodes_list[j]
            dist = compute_euclidean_tau(n1, n2)
            graph.add_edge(n1.id, n2.id, dist)
            graph.add_edge(n2.id, n1.id, dist) # Bidirectional for simplicity
    return graph

def test_calculate_route_metrics_basic(sample_graph):
    # Route: D -> C1 -> C2 -> D
    routes = [["D", "C1", "C2", "D"]]
    depot_id = "D"
    vehicle_capacity = 20

    metrics = calculate_route_metrics(sample_graph, routes, depot_id, vehicle_capacity)

    # D(0,0) C1(10,0) C2(20,0) D(0,0)
    # Distances: D-C1=10, C1-C2=10, C2-D=20
    # Total Distance = 10 + 10 + 20 = 40
    assert pytest.approx(metrics["total_distance"]) == 40.0
    assert metrics["num_vehicles"] == 1
    assert metrics["capacity_violations"] == 0
    assert metrics["time_window_violations"] == 0
    assert metrics["is_feasible"] is True

    # Simulate time for D -> C1 -> C2 -> D
    # D: e=0, s=0, l=100
    # C1: e=10, s=5, l=20, demand=10
    # C2: e=15, s=3, l=25, demand=5
    # Vehicle capacity = 20

    # D -> C1:
    # Arrive C1: 0 (start) + 10 (travel) = 10.
    # Service Start C1: max(10, C1.e=10) = 10.
    # Waiting time at C1: 0.
    # Current load: 10 (C1.demand) <= 20 (capacity). OK.
    # Finish C1: 10 (start) + 5 (service) = 15.

    # C1 -> C2:
    # Arrive C2: 15 (finish C1) + 10 (travel) = 25.
    # Service Start C2: max(25, C2.e=15) = 25.
    # Waiting time at C2: 0.
    # Current load: 10 (C1) + 5 (C2) = 15 <= 20 (capacity). OK.
    # Finish C2: 25 (start) + 3 (service) = 28.

    # C2 -> D:
    # Arrive D: 28 (finish C2) + 20 (travel) = 48.
    # Service Start D: max(48, D.e=0) = 48.
    # Waiting time at D: 48.
    # Finish D: 48 + 0 (service) = 48.
    # Total route duration: 48 - 0 = 48.

    assert pytest.approx(metrics["total_service_time"]) == 8.0 # C1.s + C2.s = 5 + 3 = 8
    assert pytest.approx(metrics["total_waiting_time"]) == 0.0 # No waiting at customers
    assert pytest.approx(metrics["total_route_duration"]) == 48.0
    assert pytest.approx(metrics["total_demand_served"]) == 15.0 # C1.demand + C2.demand = 10 + 5 = 15

def test_capacity_violation(sample_graph):
    # Route: D -> C1 -> C2 -> D
    routes = [["D", "C1", "C2", "D"]]
    depot_id = "D"
    vehicle_capacity = 12 # C1 demand 10, C2 demand 5. Total 15 > 12.

    metrics = calculate_route_metrics(sample_graph, routes, depot_id, vehicle_capacity)

    assert metrics["capacity_violations"] == 1
    assert metrics["is_feasible"] is False # Should be false due to capacity violation

def test_time_window_violation_late_arrival(sample_graph):
    # Route: D -> C1 -> C2 -> D
    routes = [["D", "C1", "C2", "D"]]
    depot_id = "D"
    vehicle_capacity = 20

    # Modify C2's time window to be very tight, causing a late arrival
    sample_graph.nodes["C2"].e = 10
    sample_graph.nodes["C2"].l = 20 # Original was 15,25. Now 10,20.
    # D-C1-C2-D simulation:
    # Arrive C2 at 25. C2.l is now 20. 25 > 20 -> violation.

    metrics = calculate_route_metrics(sample_graph, routes, depot_id, vehicle_capacity)

    assert metrics["time_window_violations"] == 1
    assert metrics["is_feasible"] is False # Should be false due to time window violation

def test_time_window_violation_depot_return_late(sample_graph):
    # Route: D -> C1 -> C2 -> D
    routes = [["D", "C1", "C2", "D"]]
    depot_id = "D"
    vehicle_capacity = 20

    # Modify depot's latest time to be very tight
    sample_graph.nodes["D"].l = 40 # Original was 100. Vehicle arrives at D at 48. 48 > 40 -> violation.

    metrics = calculate_route_metrics(sample_graph, routes, depot_id, vehicle_capacity)

    assert metrics["time_window_violations"] == 1
    assert metrics["is_feasible"] is False # Should be false due to depot return violation

def test_multiple_routes(sample_graph):
    # Routes: [D -> C1 -> D], [D -> C2 -> C3 -> D]
    routes = [["D", "C1", "D"], ["D", "C2", "C3", "D"]]
    depot_id = "D"
    vehicle_capacity = 20

    metrics = calculate_route_metrics(sample_graph, routes, depot_id, vehicle_capacity)

    assert metrics["num_vehicles"] == 2
    # Route 1 (D-C1-D): Load 10 <= 20. OK.
    # Route 2 (D-C2-C3-D): Load C2(5) + C3(20) = 25. 25 > 20 (vehicle_capacity). VIOLATION.
    assert metrics["capacity_violations"] == 1 # Corrected assertion
    assert metrics["time_window_violations"] == 0
    assert metrics["is_feasible"] is False # Should be False due to capacity violation in Route 2

    # Route 1: D -> C1 -> D
    # D(0,0) C1(10,0) D(0,0)
    # Distances: D-C1=10, C1-D=10. Total=20.
    # Time: Arrive C1: 10. Service C1: 10+5=15. Arrive D: 15+10=25. Duration=25.
    # Load: 10.

    # Route 2: D -> C2 -> C3 -> D
    # D(0,0) C2(20,0) C3(30,0) D(0,0)
    # Distances: D-C2=20, C2-C3=10, C3-D=30. Total=60.
    # Time: Arrive C2: 20. Service C2: 20+3=23.
    #       Arrive C3: 23+10=33. Service C3: 33+2=35.
    #       Arrive D: 35+30=65. Duration=65.
    # Load: 5+20=25. (This will be a capacity violation if vehicle_capacity is too low, but here it's 20, so it's fine)

    # Total Distance = 20 + 60 = 80
    assert pytest.approx(metrics["total_distance"]) == 80.0
    assert pytest.approx(metrics["total_service_time"]) == 5 + 3 + 2 # C1.s + C2.s + C3.s = 10
    assert pytest.approx(metrics["total_waiting_time"]) == 0.0 # No waiting
    assert pytest.approx(metrics["total_route_duration"]) == 25 + 65 # Sum of individual route durations = 90
    assert pytest.approx(metrics["total_demand_served"]) == 10 + 5 + 20 # C1+C2+C3 = 35

def test_empty_routes(sample_graph):
    routes = [] # No routes
    depot_id = "D"
    vehicle_capacity = 10

    metrics = calculate_route_metrics(sample_graph, routes, depot_id, vehicle_capacity)

    # For no routes, all metrics should be zero and it should be feasible.
    assert metrics["total_distance"] == 0.0
    assert metrics["total_service_time"] == 0.0
    assert metrics["total_waiting_time"] == 0.0
    assert metrics["total_route_duration"] == 0.0
    assert metrics["total_demand_served"] == 0.0
    assert metrics["time_window_violations"] == 0
    assert metrics["capacity_violations"] == 0
    assert metrics["num_vehicles"] == 0
    assert metrics["is_feasible"] is True # Corrected assertion

def test_routes_with_only_depot(sample_graph):
    routes = [["D", "D"], ["D"]] # Routes that are just depot or empty
    depot_id = "D"
    vehicle_capacity = 10

    metrics = calculate_route_metrics(sample_graph, routes, depot_id, vehicle_capacity)

    assert metrics["total_distance"] == 0.0
    assert metrics["total_service_time"] == 0.0
    assert metrics["total_waiting_time"] == 0.0
    assert metrics["total_route_duration"] == 0.0
    assert metrics["total_demand_served"] == 0.0
    assert metrics["time_window_violations"] == 0
    assert metrics["capacity_violations"] == 0
    assert metrics["num_vehicles"] == 2 # Two vehicles were dispatched, even if they did nothing
    assert metrics["is_feasible"] is True

def test_time_window_waiting_time(sample_graph):
    # Route: D -> C1 -> D
    routes = [["D", "C1", "D"]]
    depot_id = "D"
    vehicle_capacity = 20

    # Modify C1's earliest time to force waiting
    sample_graph.nodes["C1"].e = 15 # Original was 10. Arrive C1 at 10, must wait until 15.
    # D -> C1:
    # Arrive C1: 10. Service Start C1: max(10, C1.e=15) = 15.
    # Waiting time at C1: 15 - 10 = 5.

    metrics = calculate_route_metrics(sample_graph, routes, depot_id, vehicle_capacity)

    assert pytest.approx(metrics["total_waiting_time"]) == 5.0
    assert metrics["is_feasible"] is True # Waiting is not a violation

def test_single_node_route_skipped(sample_graph):
    # A route that only has one customer, but is not properly formed with depot.
    # The `calculate_route_metrics` should ensure it's properly framed with depot.
    routes = [["C1"]]
    depot_id = "D"
    vehicle_capacity = 10

    metrics = calculate_route_metrics(sample_graph, routes, depot_id, vehicle_capacity)

    # Route D -> C1 -> D
    # Distances: D-C1=10, C1-D=10. Total=20.
    # Time: Arrive C1: 10. Service C1: 10+5=15. Arrive D: 15+10=25. Duration=25.
    # Load: 10.

    assert pytest.approx(metrics["total_distance"]) == 20.0
    assert metrics["num_vehicles"] == 1
    assert metrics["is_feasible"] is True # Corrected assertion
    assert pytest.approx(metrics["total_demand_served"]) == 10.0

def test_multiple_capacity_violations(sample_graph):
    routes = [
        ["D", "C1", "C2", "D"], # C1+C2 = 15 demand
        ["D", "C3", "D"]        # C3 = 20 demand
    ]
    depot_id = "D"
    vehicle_capacity = 12 # Both routes will violate capacity

    metrics = calculate_route_metrics(sample_graph, routes, depot_id, vehicle_capacity)

    assert metrics["capacity_violations"] == 2 # Both routes violate
    assert metrics["is_feasible"] is False

def test_multiple_time_window_violations(sample_graph):
    routes = [
        ["D", "C1", "D"], # C1.l = 20. Arrive 10, service 10+5=15. OK.
        ["D", "C2", "D"]  # C2.l = 25. Arrive 20, service 20+3=23. OK.
    ]
    depot_id = "D"
    vehicle_capacity = 20

    # Force violations
    # C1: Arrive 10. Service Start 10. If C1.l = 9, then 10 > 9 -> violation.
    sample_graph.nodes["C1"].l = 9 # Original 20. Now 9.
    # C2: Arrive 20. Service Start 20. If C2.l = 19, then 20 > 19 -> violation.
    sample_graph.nodes["C2"].l = 19 # Original 25. Now 19.

    metrics = calculate_route_metrics(sample_graph, routes, depot_id, vehicle_capacity)

    assert metrics["time_window_violations"] == 2 # Both routes violate
    assert metrics["is_feasible"] is False

def test_mixed_violations(sample_graph):
    routes = [
        ["D", "C1", "D"], # Capacity violation (10 > 8)
        ["D", "C2", "D"]  # TW violation (C2.l = 19, arrive 20, service 20)
    ]
    depot_id = "D"
    vehicle_capacity = 8 # For C1 route

    # Force TW violation for C2 route
    sample_graph.nodes["C2"].l = 19 # Arrive 20, Service Start 20. 20 > 19 -> violation.

    metrics = calculate_route_metrics(sample_graph, routes, depot_id, vehicle_capacity)

    assert metrics["capacity_violations"] == 1
    assert metrics["time_window_violations"] == 1
    assert metrics["is_feasible"] is False

def test_all_customers_served_metric(sample_graph):
    # Test that total_demand_served reflects all customers in problem
    # This is not directly part of is_feasible now, but good to check.
    routes = [
        ["D", "C1", "D"],
        ["D", "C2", "D"],
        ["D", "C3", "D"]
    ]
    depot_id = "D"
    vehicle_capacity = 30 # Enough capacity for individual routes

    metrics = calculate_route_metrics(sample_graph, routes, depot_id, vehicle_capacity)
    
    # Total demand of all customers: C1(10) + C2(5) + C3(20) = 35
    assert pytest.approx(metrics["total_demand_served"]) == 35.0
    assert metrics["is_feasible"] is True # All individual routes are feasible

def test_partial_customers_served(sample_graph):
    # Only C1 and C2 are served, C3 is not.
    routes = [
        ["D", "C1", "D"],
        ["D", "C2", "D"]
    ]
    depot_id = "D"
    vehicle_capacity = 30 

    metrics = calculate_route_metrics(sample_graph, routes, depot_id, vehicle_capacity)
    
    # Total demand served: C1(10) + C2(5) = 15
    assert pytest.approx(metrics["total_demand_served"]) == 15.0
    assert metrics["is_feasible"] is True # The routes themselves are feasible, even if not all customers were served.

