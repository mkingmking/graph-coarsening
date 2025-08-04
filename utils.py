import re
import csv
import io
import logging

from graph import Graph, compute_euclidean_tau
from node import Node

logger = logging.getLogger(__name__)

def parse_float(value: str) -> float:
    """Safely parse a float from a potentially malformed string.

    The Solomon datasets occasionally contain extra whitespace or stray
    characters within numeric fields (e.g. "0.00    1").  This helper
    extracts the first numeric value it can find in the string and
    converts it to ``float``.  If no valid number is found a ``ValueError``
    is raised.
    """
    try:
        return float(value)
    except ValueError:
        match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", value)
        if match:
            return float(match.group(0))
        raise

import math
from graph import Graph, compute_euclidean_tau

def calculate_route_metrics(graph: Graph, routes: list, depot_id: str, vehicle_capacity: float) -> dict:
    """
    Calculates various metrics for a set of VRP routes.
    
    Args:
        graph (Graph): The graph containing all nodes (depot and customers).
        routes (list): A list of lists, where each inner list represents a route
                       (e.g., ['D', 'C1', 'C2', 'D']).
        depot_id (str): The ID of the depot node.
        vehicle_capacity (float): The maximum capacity of a vehicle.
        
    Returns:
        dict: A dictionary containing calculated metrics.
    """
    total_distance = 0.0
    total_service_time = 0.0
    total_waiting_time = 0.0
    total_route_duration = 0.0
    total_demand_served = 0.0
    time_window_violations = 0
    capacity_violations = 0
    num_vehicles = len(routes)
    is_feasible = True # Assume feasible initially based on *generated routes*

    customers_served_by_routes = set() # To track unique customers visited by any route

    # Get all actual customer nodes in the graph (excluding depot)
    all_customers_in_graph = {node_id for node_id in graph.nodes if node_id != depot_id}

    for route in routes:
        # Skip empty routes or routes that are just [depot, depot]
        if not route or (len(route) == 2 and route[0] == depot_id and route[1] == depot_id):
            continue

        current_time = graph.nodes[depot_id].e # Each vehicle starts at depot's earliest time window
        current_load = 0.0
        route_distance = 0.0
        route_start_time = current_time # For calculating this route's total duration

        # Ensure route starts and ends with depot for consistent simulation
        simulated_path = list(route)
        if simulated_path[0] != depot_id:
            simulated_path.insert(0, depot_id)
        if simulated_path[-1] != depot_id:
            simulated_path.append(depot_id)
        
        # Simulate each segment of the route
        for i in range(len(simulated_path) - 1):
            from_node_id = simulated_path[i]
            to_node_id = simulated_path[i+1]

            from_node = graph.nodes[from_node_id]
            to_node = graph.nodes[to_node_id]

            travel_time = compute_euclidean_tau(from_node, to_node)
            
            # Update current_time for arrival at to_node
            current_time += travel_time
            route_distance += travel_time

            # If arriving at a customer node (not depot)
            if to_node_id != depot_id:
                # Capacity check
                current_load += to_node.demand
                if current_load > vehicle_capacity:
                    capacity_violations += 1
                    is_feasible = False # Route is infeasible due to capacity

                # Time window check (arrival vs. earliest) and waiting time
                if current_time < to_node.e:
                    total_waiting_time += (to_node.e - current_time)
                    current_time = to_node.e # Wait until earliest service time
                
                # Time window check (arrival vs. latest)
                if current_time > to_node.l:
                    time_window_violations += 1
                    is_feasible = False # Route is infeasible due to time window violation
                
                # Add service time
                current_time += to_node.s
                total_service_time += to_node.s
                total_demand_served += to_node.demand
                customers_served_by_routes.add(to_node_id)
            else: # Arriving at depot (final segment of a route)
                # Check if returning to depot within its time window
                if current_time > to_node.l:
                    time_window_violations += 1
                    is_feasible = False # Route is infeasible if depot return is late

        total_distance += route_distance
        total_route_duration += (current_time - route_start_time) # Duration of this specific route

    
    metrics = {
        "total_distance": total_distance,
        "total_service_time": total_service_time,
        "total_waiting_time": total_waiting_time,
        "total_route_duration": total_route_duration,
        "total_demand_served": total_demand_served,
        "time_window_violations": time_window_violations,
        "capacity_violations": capacity_violations,
        "num_vehicles": num_vehicles,
        "is_feasible": is_feasible 
    }
    return metrics


def load_graph_from_csv(file_path: str) -> tuple[Graph, str, float]:
    """
    Loads graph data from a Solomon VRPTW CSV file.
    Assumes a specific Solomon format with header lines before the data.
    
    Args:
        file_path (str): The path to the CSV file.
        
    Returns:
        tuple: A tuple containing:
            - Graph: The loaded graph object.
            - str: The ID of the depot node.
            - float: The vehicle capacity extracted from the file.
    """
    graph = Graph()
    depot_id = None
    vehicle_capacity = None # Will be read from file

    # Define the actual column headers found in Solomon datasets
    solomon_headers = [
        'CUST NO.', 'XCOORD.', 'YCOORD.', 'DEMAND', 'READY TIME', 'DUE DATE', 'SERVICE TIME'
    ]

    try:
        with open(file_path, mode='r', newline='') as f:
            # Read all lines to parse header and then use StringIO for DictReader
            lines = f.readlines()
            
            # --- Parse Vehicle Capacity ---
            # Solomon files typically have capacity on line 4 (index 3)
            # The format is like '  25         200' or '3,45,70,30,825,870,90'
            if len(lines) >= 4:
                capacity_line = lines[3].strip()
                
                # Try to parse as comma-separated first (common for some Solomon variants)
                parts = capacity_line.split(',')
                if len(parts) >= 2:
                    try:
                        # Capacity is typically the second value in the comma-separated line (index 1)
                        vehicle_capacity = float(parts[1].strip())
                    except ValueError:
                        # If parsing as float fails, it's not the expected comma-separated format
                        pass
                
                # If not found or failed, try the space-separated regex (for other Solomon variants)
                if vehicle_capacity is None:
                    # Regex to find the second number in the line, which is usually the capacity.
                    # This pattern looks for one or more digits, followed by one or more spaces,
                    # then captures one or more digits (the capacity).
                    capacity_match = re.search(r'\s*\d+\s+(\d+\.?\d*)', capacity_line)
                    if capacity_match:
                        vehicle_capacity = float(capacity_match.group(1))
                
                if vehicle_capacity is None:
                    raise ValueError(f"Could not parse vehicle capacity from line 4: '{capacity_line}'")
            else:
                raise ValueError("File is too short to contain vehicle capacity information (expected at least 4 lines).")

            # --- Prepare data for DictReader ---
            # The actual data starts from line 10 (index 9), so we skip lines 0-8.
            # The header line is line 9 (index 8). We will explicitly provide the headers.
            if len(lines) < 10: # Data starts from line 10 (index 9)
                raise ValueError("File is too short to contain customer data.")
            
            # The actual data rows start from line 10 (index 9)
            data_lines = lines[9:] # From the first data line onwards
            data_io = io.StringIO("".join(data_lines))

            # Use DictReader with explicitly provided fieldnames and COMMA as delimiter
            reader = csv.DictReader(data_io, fieldnames=solomon_headers, delimiter=',', skipinitialspace=True)

            # --- Process Customer Data ---
            for i, row in enumerate(reader):
                # Clean row data: strip spaces from keys and values, filter out empty keys/values
                cleaned_row = {}
                for k, v in row.items():
                    if k is not None and v is not None:
                        stripped_k = k.strip()
                        stripped_v = v.strip()
                        if stripped_k != '' and stripped_v != '':
                            cleaned_row[stripped_k] = stripped_v

                # Check if the cleaned_row is empty, which can happen if a row is entirely whitespace or malformed
                if not cleaned_row:
                    continue # Skip empty rows

                try:
                    # Use the solomon_headers directly as keys for consistency with DictReader
                    node_id = cleaned_row[solomon_headers[0]] # CUST NO.
                    x = parse_float(cleaned_row[solomon_headers[1]]) # XCOORD.
                    y = parse_float(cleaned_row[solomon_headers[2]]) # YCOORD.
                    demand = parse_float(cleaned_row[solomon_headers[3]]) # DEMAND
                    e = parse_float(cleaned_row[solomon_headers[4]]) # READY TIME
                    l = parse_float(cleaned_row[solomon_headers[5]]) # DUE DATE
                    s = parse_float(cleaned_row[solomon_headers[6]]) # SERVICE TIME
                    
                    node = Node(node_id, x, y, s, e, l, demand)
                    graph.add_node(node)
                    
                    if i == 0: # The first node in the data section is the depot
                        depot_id = node_id
                except (ValueError, KeyError) as data_error:
                    # Provide more context for data parsing errors
                    raise ValueError(f"Error processing data in row {i+1} of {file_path}. Row content: {cleaned_row}. Details: {data_error}") from data_error
                
        if depot_id is None:
            raise ValueError("No nodes found in CSV data or depot not identified.")
        if vehicle_capacity is None:
            raise ValueError("Vehicle capacity could not be determined from the file.")

        # Add edges between all nodes (assuming a complete graph for simplicity)
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



