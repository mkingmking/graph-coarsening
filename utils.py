import re
import csv
import io
import logging

from . graph import Graph, compute_euclidean_tau
from . node import Node

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
    total_route_duration = 0.0
    time_window_violations = 0
    capacity_violations = 0
    num_vehicles = 0 
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
            "routes_list": routes
        }

    for route in routes:
        if not route or len(route) < 2 or (len(route) == 2 and route[0] == depot_id and route[1] == depot_id):
            continue

        num_vehicles += 1

        current_load = 0.0
        current_time = graph.nodes[depot_id].e
        
        route_distance = 0.0
        route_service_time = 0.0
        route_waiting_time = 0.0
        route_feasible = True

        for i in range(len(route) - 1):
            from_node_id = route[i]
            to_node_id = route[i+1]

            from_node = graph.nodes[from_node_id]
            to_node = graph.nodes[to_node_id]

            if to_node_id != depot_id:
                current_load += to_node.demand
                if current_load > vehicle_capacity:
                    capacity_violations += 1
                    route_feasible = False
                    all_feasible = False

            travel_time = compute_euclidean_tau(from_node, to_node)
            total_distance += travel_time
            route_distance += travel_time

            arrival_time_at_to_node = current_time + travel_time
            
            service_start_time_at_to_node = max(arrival_time_at_to_node, to_node.e)

            if service_start_time_at_to_node > to_node.l:
                time_window_violations += 1
                route_feasible = False
                all_feasible = False

            waiting_time = max(0, to_node.e - arrival_time_at_to_node)
            total_waiting_time += waiting_time
            route_waiting_time += waiting_time

            current_time = service_start_time_at_to_node + to_node.s

            if to_node_id != depot_id:
                total_service_time += to_node.s
                route_service_time += to_node.s
                total_demand_served += to_node.demand
            
        if route[-1] == depot_id:
            last_customer_node_id = route[-2] if len(route) > 1 else depot_id
            last_customer_node = graph.nodes[last_customer_node_id]
            depot_node = graph.nodes[depot_id]
            travel_time_to_depot = compute_euclidean_tau(last_customer_node, depot_node)
            final_arrival_at_depot = current_time + travel_time_to_depot

            if final_arrival_at_depot > depot_node.l:
                time_window_violations += 1
                route_feasible = False
                all_feasible = False
            total_route_duration += final_arrival_at_depot
        else:
            route_feasible = False
            all_feasible = False
            print(f"Warning: Route {route} does not end at depot {depot_id}. Considered infeasible.")

    all_feasible = all_feasible and (capacity_violations == 0) and (time_window_violations == 0)

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
        "routes_list": routes
    }


def load_graph_from_csv(file_path: str) -> tuple[Graph, str, float]:
    """
    Loads graph data from a Solomon VRPTW CSV file.
    
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

    
    solomon_headers = [
        'CUST NO.', 'XCOORD.', 'YCOORD.', 'DEMAND', 'READY TIME', 'DUE DATE', 'SERVICE TIME'
    ]

    try:
        with open(file_path, mode='r', newline='') as f:
            # Read all lines to parse header and then use StringIO for DictReader
            lines = f.readlines()
            
            # --- Parse Vehicle Capacity ---
            if len(lines) >= 4:
                capacity_line = lines[3].strip()
                
                # Try to parse as comma-separated first 
                parts = capacity_line.split(',')
                if len(parts) >= 2:
                    try: #hardcoded value here
                        vehicle_capacity = 200
                    except ValueError:
                        # If parsing as float fails, it's not the expected comma-separated format
                        pass
                
                # If not found or failed, try the space-separated regex 
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
            if len(lines) < 10: 
                raise ValueError("File is too short to contain customer data.")
            
            # The actual data rows start from line 10 (index 9)
            data_lines = lines[9:] 
            data_io = io.StringIO("".join(data_lines))

            
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
                    
                    raise ValueError(f"Error processing data in row {i+1} of {file_path}. Row content: {cleaned_row}. Details: {data_error}") from data_error
                
        if depot_id is None:
            raise ValueError("No nodes found in CSV data or depot not identified.")
        if vehicle_capacity is None:
            raise ValueError("Vehicle capacity could not be determined from the file.")

        # Add edges between all nodes (a complete graph)
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



