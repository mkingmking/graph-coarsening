import re
import csv
import io
import logging
from pathlib import Path

from .graph import Graph, compute_euclidean_tau
from .node import Node

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Reference total demand served for every Solomon benchmark instance.
# Values are taken from the OR-Tools baseline run (100 % customer coverage,
# all solutions feasible).  Used by check_coverage() to detect solvers that
# silently drop customers.
# ---------------------------------------------------------------------------
_ORTOOLS_DEMAND: dict[str, float] = {
    "C101": 1810.0, "C102": 1810.0, "C103": 1810.0, "C104": 1810.0,
    "C105": 1810.0, "C106": 1810.0, "C107": 1810.0, "C108": 1810.0,
    "C109": 1810.0, "C201": 1810.0, "C202": 1810.0, "C203": 1810.0,
    "C204": 1810.0, "C205": 1810.0, "C206": 1810.0, "C207": 1810.0,
    "C208": 1810.0,
    "R101": 1458.0, "R102": 1458.0, "R103": 1458.0, "R104": 1458.0,
    "R105": 1458.0, "R106": 1458.0, "R107": 1458.0, "R108": 1458.0,
    "R109": 1458.0, "R110": 1458.0, "R111": 1458.0, "R112": 1458.0,
    "R201": 1458.0, "R202": 1458.0, "R203": 1458.0, "R204": 1458.0,
    "R205": 1458.0, "R206": 1458.0, "R207": 1458.0, "R208": 1458.0,
    "R209": 1458.0, "R210": 1458.0, "R211": 1458.0,
    "RC101": 1724.0, "RC102": 1724.0, "RC103": 1724.0, "RC104": 1724.0,
    "RC105": 1724.0, "RC106": 1724.0, "RC107": 1724.0, "RC108": 1724.0,
    "RC201": 1724.0, "RC202": 1724.0, "RC203": 1724.0, "RC204": 1724.0,
    "RC205": 1724.0, "RC206": 1724.0, "RC207": 1724.0, "RC208": 1724.0,
}


# ---------------------------------------------------------------------------
# Per-family coarsening hyperparameters.
# Keys match the Solomon family folder names (C1, C2, R1, R2, RC1, RC2).
# These are the best configurations found via random hyperparameter search;
# leave RC entries as the default until the RC search is complete.
# ---------------------------------------------------------------------------
FAMILY_HYPERPARAMS: dict[str, dict] = {
    "C1":  {"alpha": 1.0, "beta": 1.0, "P": 0.5, "radiusCoeff": 2.0},
    "C2":  {"alpha": 1.0, "beta": 1.0, "P": 0.5, "radiusCoeff": 2.0},
    "R1":  {"alpha": 1.0, "beta": 0.6, "P": 0.4, "radiusCoeff": 0.5},
    "R2":  {"alpha": 1.0, "beta": 0.6, "P": 0.4, "radiusCoeff": 0.5},
    "RC1": {"alpha": 1.0, "beta": 1.0, "P": 0.5, "radiusCoeff": 2.0},  # placeholder
    "RC2": {"alpha": 1.0, "beta": 1.0, "P": 0.5, "radiusCoeff": 2.0},  # placeholder
}

DEFAULT_HYPERPARAMS: dict = {"alpha": 1.0, "beta": 1.0, "P": 0.5, "radiusCoeff": 2.0}


def detect_family(file_path: str) -> str | None:
    """Return the Solomon family (e.g. 'R1', 'C2') from a file path, or None."""
    for family in ("RC1", "RC2", "C1", "C2", "R1", "R2"):  # RC before C/R to avoid prefix clash
        if f"/{family}/" in file_path or f"\\{family}\\" in file_path:
            return family
    return None


def resolve_coarsening_params(
    file_path: str,
    alpha: float | None = None,
    beta: float | None = None,
    P: float | None = None,
    radiusCoeff: float | None = None,
) -> dict:
    """Return the coarsening hyperparameter dict to use for *file_path*.

    Resolution order (highest priority first):
    1. Any explicitly supplied keyword argument overrides the corresponding param.
    2. If the file belongs to a known Solomon family, use that family's defaults.
    3. Otherwise fall back to DEFAULT_HYPERPARAMS.
    """
    family = detect_family(file_path)
    base = FAMILY_HYPERPARAMS.get(family, DEFAULT_HYPERPARAMS).copy()
    if alpha is not None:
        base["alpha"] = alpha
    if beta is not None:
        base["beta"] = beta
    if P is not None:
        base["P"] = P
    if radiusCoeff is not None:
        base["radiusCoeff"] = radiusCoeff
    return base


def check_coverage(instance_name: str, metrics: dict) -> bool:
    """Check that a solver's solution serves every customer.

    Compares ``metrics['total_demand_served']`` against the OR-Tools
    reference value stored in ``_ORTOOLS_DEMAND``.

    Parameters
    ----------
    instance_name:
        Solomon benchmark name, e.g. ``"C101"`` or ``"RC205"``.
        Case-insensitive; leading path components and ``.csv`` suffix
        are stripped automatically so callers may pass a raw file path.
    metrics:
        Dict returned by ``calculate_route_metrics``.

    Returns
    -------
    bool
        ``True`` if demand served matches the reference (full coverage),
        ``False`` otherwise.  Always returns ``True`` when the instance is
        not in the reference dictionary (unknown benchmark) so that callers
        are not blocked on unsupported instances.
    """
    # Normalise: strip path and extension, upper-case
    key = Path(instance_name).stem.upper()

    expected = _ORTOOLS_DEMAND.get(key)
    if expected is None:
        logger.warning(
            f"check_coverage: '{key}' not in reference dictionary — skipping check."
        )
        return True

    served = metrics.get("total_demand_served", 0.0)
    if abs(served - expected) > 1e-6:
        pct = served / expected * 100
        logger.error(
            f"COVERAGE FAILURE [{key}]: served {served:.1f} / {expected:.1f} "
            f"({pct:.1f}%) — {expected - served:.1f} demand units dropped."
        )
        return False

    logger.info(f"check_coverage [{key}]: OK ({served:.1f} / {expected:.1f})")
    return True

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
        depot_arrival_time = None  # set when the closing depot is reached in the loop

        for i in range(len(route) - 1):
            from_node_id = route[i]
            to_node_id = route[i+1]

            from_node = graph.nodes[from_node_id]
            to_node = graph.nodes[to_node_id]

            if to_node_id != depot_id:
                current_load += to_node.demand
                if current_load > vehicle_capacity:
                    capacity_violations += 1
                    all_feasible = False

            travel_time = compute_euclidean_tau(from_node, to_node)
            total_distance += travel_time

            arrival_time_at_to_node = current_time + travel_time
            service_start_time_at_to_node = max(arrival_time_at_to_node, to_node.e)

            if service_start_time_at_to_node > to_node.l:
                time_window_violations += 1
                all_feasible = False

            waiting_time = max(0, to_node.e - arrival_time_at_to_node)
            total_waiting_time += waiting_time

            current_time = service_start_time_at_to_node + to_node.s

            if to_node_id != depot_id:
                total_service_time += to_node.s
                total_demand_served += to_node.demand
            else:
                # Record actual arrival time at closing depot (before service).
                # This is used for route duration; do NOT recompute after the
                # loop — that would double-count the travel time.
                depot_arrival_time = arrival_time_at_to_node

        if route[-1] == depot_id:
            if depot_arrival_time is not None:
                total_route_duration += depot_arrival_time
            # TW check for the closing depot was already performed inside the
            # loop above (service_start_time_at_to_node > to_node.l). No
            # re-check here — that would double-count violations.
        else:
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


"""def load_graph_from_csv(file_path: str) -> tuple[Graph, str, float]:
    
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
"""


def _solomon_capacity(file_path: str) -> float | None:
    """
    Returns the known vehicle capacity for a Solomon benchmark instance based on
    the filename, ignoring whatever the CSV header may (or may not) contain.

    Families and their capacities:
      C1  (C101–C109)   : 200
      C2  (C201–C208)   : 700
      R1  (R101–R112)   : 200
      R2  (R201–R211)   : 1000
      RC1 (RC101–RC108) : 200
      RC2 (RC201–RC208) : 1000

    Returns None if the filename does not match any known Solomon pattern,
    leaving the caller to use the value parsed from the file.
    """
    stem = Path(file_path).stem.lower()  # e.g. "c101", "rc208"

    _CAPACITY_MAP = {
        'c1': 200.0,
        'c2': 700.0,
        'r1': 200.0,
        'r2': 1000.0,
        'rc1': 200.0,
        'rc2': 1000.0,
    }

    for prefix, cap in _CAPACITY_MAP.items():
        if stem.startswith(prefix) and len(stem) > len(prefix) and stem[len(prefix)].isdigit():
            return cap

    return None


def load_graph_from_csv(file_path: str) -> tuple[Graph, str, float]:
    """
    Loads graph data from a Solomon VRPTW CSV file.
    ROBUST VERSION: Automatically detects start of data.

    Vehicle capacity is resolved exclusively from the Solomon family map
    (_solomon_capacity). The CSV header is never used for capacity.
    Raises ValueError if the filename does not match a known Solomon instance.
    """
    vehicle_capacity = _solomon_capacity(file_path)
    if vehicle_capacity is None:
        raise ValueError(
            f"Unknown Solomon instance '{Path(file_path).stem}': cannot determine vehicle capacity. "
            "Add the instance to _solomon_capacity() if it is a new benchmark family."
        )

    graph = Graph()
    depot_id = None

    solomon_headers = [
        'CUST NO.', 'XCOORD.', 'YCOORD.', 'DEMAND', 'READY TIME', 'DUE DATE', 'SERVICE TIME'
    ]

    try:
        with open(file_path, mode='r', newline='') as f:
            lines = f.readlines()

            # --- 1. Robust Header Detection ---
            header_index = -1
            for idx, line in enumerate(lines):
                if 'CUST NO.' in line:
                    header_index = idx
                    break

            if header_index == -1:
                raise ValueError("Could not find 'CUST NO.' header row in file.")

            # --- 2. Parse Data ---
            # Data starts immediately after the header line
            data_lines = lines[header_index + 1:] 
            data_io = io.StringIO("".join(data_lines))
            
            # We use the known fieldnames to ensure correct mapping
            reader = csv.DictReader(data_io, fieldnames=solomon_headers, delimiter=',', skipinitialspace=True)

            for i, row in enumerate(reader):
                # Clean row data
                cleaned_row = {}
                for k, v in row.items():
                    if k and v and k.strip() and v.strip():
                        cleaned_row[k.strip()] = v.strip()

                if not cleaned_row: continue

                try:
                    # Parse using solomon_headers keys
                    # (Your CSV might have headers, but we force our standard keys for access)
                    # Note: DictReader with fieldnames maps the columns in order.
                    # Since we skipped the header line, the first data line maps to these keys.
                    
                    node_id = cleaned_row[solomon_headers[0]] # CUST NO.
                    x = parse_float(cleaned_row[solomon_headers[1]])
                    y = parse_float(cleaned_row[solomon_headers[2]])
                    demand = parse_float(cleaned_row[solomon_headers[3]])
                    e = parse_float(cleaned_row[solomon_headers[4]])
                    l = parse_float(cleaned_row[solomon_headers[5]])
                    s = parse_float(cleaned_row[solomon_headers[6]])
                    
                    node = Node(node_id, x, y, s, e, l, demand)
                    graph.add_node(node)
                    
                    # The first node read is the depot
                    if depot_id is None:
                        depot_id = node_id
                        
                except (ValueError, KeyError) as e:
                    # Skip lines that might be malformed or empty
                    continue
                
        if depot_id is None:
            raise ValueError("No nodes found in CSV data.")

        # Add complete graph edges
        node_ids = list(graph.nodes.keys())
        for i in range(len(node_ids)):
            for j in range(i + 1, len(node_ids)):
                id1, id2 = node_ids[i], node_ids[j]
                node1, node2 = graph.nodes[id1], graph.nodes[id2]
                tau = compute_euclidean_tau(node1, node2)
                graph.add_edge(id1, id2, tau)

        logger.info(f"Successfully loaded graph. Depot ID: {depot_id}, Capacity: {vehicle_capacity}")
        return graph, depot_id, vehicle_capacity

    except Exception as e:
        logger.error(f"Error loading graph: {e}")
        import traceback
        traceback.print_exc()
        raise


