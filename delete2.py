import re
import csv
import io
import logging

from .graph import Graph, compute_euclidean_tau
from .node import Node

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



def calculate_route_metrics(graph: Graph, routes: list, depot_id: str, vehicle_capacity: float) -> dict:
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
    total_duration = 0.0
    num_vehicles = len(routes)
    
    tw_violations = 0
    cap_violations = 0

    for route in routes:
        if not route or len(route) < 2:
            continue
            
        current_distance = 0.0
        current_time = 0.0
        current_capacity = 0.0

        for i in range(len(route) - 1):
            current_node_id = route[i]
            next_node_id = route[i+1]
            
            current_node = graph.nodes[current_node_id]
            next_node = graph.nodes[next_node_id]

            # Travel time/distance
            travel_time = compute_euclidean_tau(current_node, next_node)
            current_distance += travel_time
            current_time += travel_time

            # Update for next node
            if current_node_id != depot_id:
                # Add service time from the *previous* node
                current_time += current_node.s
                total_service_time += current_node.s

            # Time window and waiting time
            if current_time < next_node.e:
                waiting_time = next_node.e - current_time
                current_time = next_node.e
                total_waiting_time += waiting_time
            
            # Check time window violation
            if current_time > next_node.l:
                tw_violations += 1
            
            # Update capacity
            if next_node_id != depot_id:
                current_capacity += next_node.demand

        # Add service time for the last node before returning to depot
        if route[-1] != depot_id:
            current_time += graph.nodes[route[-1]].s
            total_service_time += graph.nodes[route[-1]].s

        total_distance += current_distance
        total_duration += current_time
        
        # Check capacity violation for the entire route
        if current_capacity > vehicle_capacity:
            cap_violations += 1

    is_feasible = (tw_violations == 0 and cap_violations == 0)
    
    return {
        "total_distance": total_distance,
        "num_vehicles": num_vehicles,
        "total_duration": total_duration,
        "total_service_time": total_service_time,
        "total_waiting_time": total_waiting_time,
        "tw_violations_count": tw_violations,
        "cap_violations_count": cap_violations,
        "feasible": is_feasible
    }


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
            lines = f.readlines()
            
            # --- Parse Vehicle Capacity ---
            # Solomon files typically have capacity on line 4 (index 3).
            # The format is often space-separated, e.g., '  25         200'.
            # Some variants may be comma-separated. We'll try both.
            if len(lines) >= 4:
                capacity_line = lines[3].strip()
                
                # Try to parse space-separated format first
                parts = capacity_line.split()
                if len(parts) >= 2:
                    try:
                        # Capacity is typically the second value in the space-separated line
                        vehicle_capacity = float(parts[1].strip())
                    except (ValueError, IndexError):
                        pass

                # If not found or failed, try comma-separated
                if vehicle_capacity is None:
                    parts = capacity_line.split(',')
                    if len(parts) >= 2:
                        try:
                            # Capacity is typically the second value in the comma-separated line
                            vehicle_capacity = float(parts[1].strip())
                        except (ValueError, IndexError):
                            pass
                
                if vehicle_capacity is None:
                    # Fallback to hardcoded value if parsing fails
                    logger.warning("Could not parse vehicle capacity from file. Using default value 200.")
                    vehicle_capacity = 200.0
            else:
                raise ValueError("File is too short to contain vehicle capacity information (expected at least 4 lines).")

            # --- Prepare data for DictReader ---
            # The actual data starts from line 10 (index 9), so we skip lines 0-8.
            # The header line is line 9 (index 8). We will explicitly provide the headers.
            if len(lines) < 10: # Data starts from line 10 (index 9)
                raise ValueError("File is too short to contain customer data.")
            
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
