# utils.py

import csv
import os
import re
import io
import logging
from graph import Graph, compute_euclidean_tau  # Import Graph and helper
from node import Node  # Import Node


def parse_float(value: str) -> float:
    """Safely parse a float from a potentially malformed string."""
    try:
        return float(value)
    except ValueError:
        match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", value)
        if match:
            return float(match.group(0))
        raise

logger = logging.getLogger(__name__)

def calculate_route_metrics(graph: Graph, routes: list, depot_id: str, vehicle_capacity: float):
    """
    Calculates various metrics for a list of routes on a specified graph.
    
    Args:
        graph (Graph): The graph (original or coarsened) on which the routes exist.
        routes (list): A list of lists of node IDs, where each inner list is a route.
        depot_id (str): The ID of the depot node.
        vehicle_capacity (float): The maximum capacity of a vehicle.
        
    Returns:
        dict: A dictionary containing aggregated calculated metrics.
    """
    total_distance = 0.0
    total_service_time = 0.0
    total_waiting_time = 0.0
    total_route_duration = 0.0 # Sum of individual route durations
    time_window_violations = 0
    capacity_violations = 0
    num_vehicles = len(routes)
    all_feasible = True
    total_demand_served = 0.0

    if not routes:
        return {
            "total_distance": 0.0,
            "total_service_time": 0.0,
            "total_waiting_time": 0.0,
            "total_route_duration": 0.0,
            "time_window_violations": 0,
            "capacity_violations": 0,
            "is_feasible": False,
            "num_vehicles": 0,
            "total_demand_served": 0.0,
            "routes_list": routes # For debugging, if needed
        }

    for route in routes:
        if not route or len(route) < 2:
            continue # Skip empty or invalid routes

        current_time = graph.nodes[depot_id].e # Each vehicle starts at depot's earliest time
        current_load = 0.0
        
        route_distance = 0.0
        route_service_time = 0.0
        route_waiting_time = 0.0
        route_feasible = True # Feasibility for this specific route

        for i in range(len(route) - 1):
            from_node_id = route[i]
            to_node_id = route[i+1]

            from_node = graph.nodes[from_node_id]
            to_node = graph.nodes[to_node_id]

            # Capacity check (only for customer nodes)
            if to_node_id != depot_id:
                current_load += to_node.demand
                if current_load > vehicle_capacity:
                    capacity_violations += 1
                    route_feasible = False
                    all_feasible = False
                    # print(f"  Violation: Route exceeds capacity at node {to_node_id}. Current load {current_load:.2f} > Capacity {vehicle_capacity:.2f}")

            # Travel time between current and next node
            travel_time = compute_euclidean_tau(from_node, to_node)
            total_distance += travel_time # Accumulate to total distance
            route_distance += travel_time

            # Arrival time at the next node
            arrival_time_at_to_node = current_time + travel_time
            
            # Service start time at the next node
            service_start_time_at_to_node = max(arrival_time_at_to_node, to_node.e)

            # Check for time window violation (service starts too late)
            if service_start_time_at_to_node > to_node.l:
                time_window_violations += 1
                route_feasible = False # This specific route is not feasible
                all_feasible = False # Overall solution is not feasible
                # print(f"  Violation: Node {to_node_id} service starts too late. Expected by {to_node.l:.2f}, service start {service_start_time_at_to_node:.2f}")

            # Calculate waiting time
            waiting_time = max(0, to_node.e - arrival_time_at_to_node)
            total_waiting_time += waiting_time # Accumulate to total waiting time
            route_waiting_time += waiting_time

            # Update current time after service at the current node
            current_time = service_start_time_at_to_node + to_node.s

            # Accumulate service time for customer nodes (not depot on return)
            if to_node_id != depot_id: # Assuming depot service time is 0 and not counted in total_service_time
                total_service_time += to_node.s # Accumulate to total service time
                route_service_time += to_node.s
                total_demand_served += to_node.demand
            
        # After a route is completed, add its duration to the total
        total_route_duration += current_time # The time when this vehicle finishes its route

    return {
        "total_distance": total_distance,
        "total_service_time": total_service_time,
        "total_waiting_time": total_waiting_time,
        "total_route_duration": total_route_duration,
        "time_window_violations": time_window_violations,
        "capacity_violations": capacity_violations,
        "is_feasible": all_feasible,
        "num_vehicles": num_vehicles,
        "total_demand_served": total_demand_served,
        "routes_list": routes # Return the list of routes for inspection
    }

def load_graph_from_csv(file_path: str) -> tuple[Graph, str, float]:
    """Loads graph data from a Solomon VRPTW CSV file."""
    graph = Graph()
    depot_id = None
    vehicle_capacity = None

    solomon_headers = [
        'CUST NO.', 'XCOORD.', 'YCOORD.', 'DEMAND',
        'READY TIME', 'DUE DATE', 'SERVICE TIME'
    ]

    try:
        with open(file_path, mode='r', newline='') as f:
            lines = f.readlines()

            if len(lines) >= 4:
                capacity_line = lines[3].strip()

                parts = capacity_line.split(',')
                if len(parts) >= 2:
                    try:
                        vehicle_capacity = float(parts[1].strip())
                    except ValueError:
                        pass

                if vehicle_capacity is None:
                    capacity_match = re.search(r"\s*\d+\s+(\d+\.?\d*)", capacity_line)
                    if capacity_match:
                        vehicle_capacity = float(capacity_match.group(1))

                if vehicle_capacity is None:
                    raise ValueError(f"Could not parse vehicle capacity from line 4: '{capacity_line}'")
            else:
                raise ValueError("File is too short to contain vehicle capacity information (expected at least 4 lines).")

            if len(lines) < 10:
                raise ValueError("File is too short to contain customer data.")

            data_lines = lines[9:]
            data_io = io.StringIO("".join(data_lines))
            reader = csv.DictReader(data_io, fieldnames=solomon_headers, delimiter=',', skipinitialspace=True)

            for i, row in enumerate(reader):
                cleaned_row = {}
                for k, v in row.items():
                    if k is not None and v is not None:
                        sk = k.strip()
                        sv = v.strip()
                        if sk != '' and sv != '':
                            cleaned_row[sk] = sv

                if not cleaned_row:
                    continue

                try:
                    node_id = cleaned_row[solomon_headers[0]]
                    x = parse_float(cleaned_row[solomon_headers[1]])
                    y = parse_float(cleaned_row[solomon_headers[2]])
                    demand = parse_float(cleaned_row[solomon_headers[3]])
                    e = parse_float(cleaned_row[solomon_headers[4]])
                    l = parse_float(cleaned_row[solomon_headers[5]])
                    s = parse_float(cleaned_row[solomon_headers[6]])

                    node = Node(node_id, x, y, s, e, l, demand)
                    graph.add_node(node)

                    if i == 0:
                        depot_id = node_id
                except (ValueError, KeyError) as data_error:
                    raise ValueError(
                        f"Error processing data in row {i+1} of {file_path}. Row content: {cleaned_row}. Details: {data_error}"
                    ) from data_error

        if depot_id is None:
            raise ValueError("No nodes found in CSV data or depot not identified.")
        if vehicle_capacity is None:
            raise ValueError("Vehicle capacity could not be determined from the file.")

        node_ids = list(graph.nodes.keys())
        for i in range(len(node_ids)):
            for j in range(i + 1, len(node_ids)):
                id1 = node_ids[i]
                id2 = node_ids[j]
                node1 = graph.nodes[id1]
                node2 = graph.nodes[id2]
                tau = compute_euclidean_tau(node1, node2)
                graph.add_edge(id1, id2, tau)

        logger.info(f"Successfully loaded graph from {file_path}. Depot ID: {depot_id}, Vehicle Capacity: {vehicle_capacity}")
        return graph, depot_id, vehicle_capacity

    except FileNotFoundError:
        logger.error(f"Error: CSV file not found at {file_path}")
        raise
    except ValueError as e:
        logger.error(f"Error processing CSV data: {e}")
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred while loading CSV: {e}")
        import traceback
        traceback.print_exc()
        raise
