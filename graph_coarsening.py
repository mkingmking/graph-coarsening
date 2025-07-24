import math
import copy
import csv
import os  # Import os module for path manipulation
import re  # Import regex for parsing capacity line
import io  # Import io for StringIO
import logging

logger = logging.getLogger(__name__)

# --- 1. Node Class ---
class Node:
    """
    Represents a customer or depot node in the VRPTW graph.
    Attributes:
        id (str): Unique identifier for the node.
        x (float): X-coordinate.
        y (float): Y-coordinate.
        s (float): Service time required at this node.
        e (float): Earliest service time (start of time window).
        l (float): Latest service time (end of time window).
        t (float): Central time, calculated as (e + (l - s)) / 2.
        demand (float): Demand of the customer at this node (0 for depot).
        is_super_node (bool): True if this node is a merged super-node.
        original_nodes (list): List of original node IDs that form this super-node.
    """
    def __init__(self, id, x, y, s, e, l, demand, is_super_node=False, original_nodes=None):
        self.id = id
        self.x = x
        self.y = y
        self.s = s
        self.e = e
        self.l = l
        self.demand = demand
        # Calculate central time (t_i) as per the proposal
        self.t = (e + (l - s)) / 2 if (l - s) >= 0 else e # Handle cases where l < s, though typically l >= s
        self.is_super_node = is_super_node
        self.original_nodes = original_nodes if original_nodes is not None else [id]

    def __repr__(self):
        return (f"Node(ID={self.id}, Coords=({self.x:.2f},{self.y:.2f}), S={self.s:.2f}, "
                f"TW=[{self.e:.2f},{self.l:.2f}], T={self.t:.2f}, Demand={self.demand:.2f}, Super={self.is_super_node})")

# --- 2. Edge Class ---
class Edge:
    """
    Represents an edge between two nodes in the graph.
    Attributes:
        u_id (str): ID of the first node.
        v_id (str): ID of the second node.
        tau (float): Euclidean travel time between u and v.
        D_ij (float): Spatio-temporal distance (weight) for this edge.
    """
    def __init__(self, u_id, v_id, tau):
        self.u_id = u_id
        self.v_id = v_id
        self.tau = tau
        self.D_ij = 0.0 # Will be computed later

    def __repr__(self):
        return f"Edge({self.u_id}-{self.v_id}, Tau={self.tau:.2f}, D_ij={self.D_ij:.2f})"

# --- 3. Graph Class ---
class Graph:
    """
    Represents the graph with nodes and edges.
    Attributes:
        nodes (dict): Dictionary mapping node ID to Node object.
        edges (list): List of Edge objects.
        adj (dict): Adjacency list mapping node ID to a set of connected node IDs.
    """
    def __init__(self):
        self.nodes = {}
        self.edges = []
        self.adj = {} # Adjacency list: {node_id: {neighbor_id, ...}}

    def add_node(self, node):
        """Adds a node to the graph."""
        self.nodes[node.id] = node
        if node.id not in self.adj:
            self.adj[node.id] = set()

    def add_edge(self, u_id, v_id, tau):
        """Adds an edge to the graph, connecting two existing nodes."""
        if u_id not in self.nodes or v_id not in self.nodes:
            raise ValueError(f"Nodes {u_id} or {v_id} not found in graph.")
        
        # Check if edge already exists to avoid duplicates
        for edge in self.edges:
            if (edge.u_id == u_id and edge.v_id == v_id) or \
               (edge.u_id == v_id and edge.v_id == u_id):
                return # Edge already exists

        edge = Edge(u_id, v_id, tau)
        self.edges.append(edge)
        self.adj[u_id].add(v_id)
        self.adj[v_id].add(u_id) # Assuming undirected graph for VRP connections

    def remove_node(self, node_id):
        """Removes a node and all its incident edges from the graph."""
        if node_id not in self.nodes:
            return

        # Remove edges connected to this node
        self.edges = [edge for edge in self.edges if edge.u_id != node_id and edge.v_id != node_id]

        # Remove from adjacency list
        if node_id in self.adj:
            for neighbor_id in list(self.adj[node_id]): # Iterate over a copy
                if neighbor_id in self.adj:
                    self.adj[neighbor_id].discard(node_id)
            del self.adj[node_id]
        
        del self.nodes[node_id]

    def get_edge_by_nodes(self, u_id, v_id):
        """Returns an edge object given its two node IDs, or None if not found."""
        for edge in self.edges:
            if (edge.u_id == u_id and edge.v_id == v_id) or \
               (edge.u_id == v_id and edge.v_id == u_id):
                return edge
        return None

    def get_neighbors(self, node_id):
        """Returns a set of neighbor IDs for a given node."""
        return self.adj.get(node_id, set())

    def get_all_edges_for_node(self, node_id):
        """Returns a list of edge objects connected to a given node."""
        return [edge for edge in self.edges if edge.u_id == node_id or edge.v_id == node_id]


# --- Helper function to compute Euclidean distance (travel time) ---
def compute_euclidean_tau(node1: Node, node2: Node) -> float:
    """
    Computes the Euclidean travel time (distance) between two nodes.
    """
    return math.sqrt((node1.x - node2.x)**2 + (node1.y - node2.y)**2)

# --- Robust float parsing helper ---
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

# --- Helper function to calculate route metrics ---
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
                    # logger.info(f"  Violation: Route exceeds capacity at node {to_node_id}. Current load {current_load:.2f} > Capacity {vehicle_capacity:.2f}")

            # Travel time between current and next node
            travel_time = compute_euclidean_tau(from_node, to_node)
            total_distance += travel_time # Accumulate to total distance
            route_distance += travel_time

            # Arrival time at the next node
            arrival_time_at_to_node = current_time + travel_time
            
            # Service start time at the next node
            service_start_time_at_to_node = max(arrival_time_at_to_node, to_node.e)

            # Check for time window violation (arriving too late)
            if service_start_time_at_to_node > to_node.l:
                time_window_violations += 1
                route_feasible = False # This specific route is not feasible
                all_feasible = False # Overall solution is not feasible
                # logger.info(f"  Violation: Node {to_node_id} arrived too late. Expected by {to_node.l:.2f}, arrived at {arrival_time_at_to_node:.2f}, service start {service_start_time_at_to_node:.2f}")

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


# --- 4. SpatioTemporalGraphCoarsener Class ---
class SpatioTemporalGraphCoarsener:
    """
    Implements the multilevel spatio-temporal graph coarsening algorithm.
    """
    def __init__(self, graph: Graph, alpha: float, beta: float, P: float, radiusCoeff: float, depot_id: str):
        self.graph = graph # This is the initial graph, it will not be modified directly
        self.alpha = alpha
        self.beta = beta
        self.P = P
        self.radiusCoeff = radiusCoeff
        self.depot_id = depot_id
        self.merge_layers = [] # Stores (super_node_id, original_node_i_id, original_node_j_id, pi_order)

    def _compute_D_ij(self, current_graph: Graph, edge: Edge) -> float:
        """
        Computes the spatio-temporal distance metric D_ij for an edge.
        D_ij = alpha * tau_ij + beta * max(0, e_j - (t_i + s_i + tau_ij))
        Takes the current_graph to ensure nodes are from the correct graph state.
        """
        node_i = current_graph.nodes[edge.u_id]
        node_j = current_graph.nodes[edge.v_id]

        # Ensure correct order for temporal slack calculation if edge is (j,i)
        if edge.u_id == node_i.id: # u is i, v is j
            tau_ij = compute_euclidean_tau(node_i, node_j) # Use the global helper
            temporal_slack = max(0, node_j.e - (node_i.t + node_i.s + tau_ij))
        else: # u is j, v is i (swap roles for calculation)
            tau_ij = compute_euclidean_tau(node_j, node_i) # Use the global helper
            temporal_slack = max(0, node_i.e - (node_j.t + node_j.s + tau_ij))
        
        return self.alpha * tau_ij + self.beta * temporal_slack

    def _evaluate_feasibility(self, current_graph: Graph, node_i: Node, node_j: Node) -> tuple[bool, bool]:
        """
        Evaluates merge feasibility for i -> j and j -> i.
        Returns (feas_i_to_j, feas_j_to_i)
        Takes the current_graph to ensure nodes are from the correct graph state.
        """
        tau_ij = compute_euclidean_tau(node_i, node_j) # Use the global helper

        # feas_i_to_j = [e_i <= l_j - s_j - tau_ij - s_i]
        feas_i_to_j = (node_i.e <= node_j.l - node_j.s - tau_ij - node_i.s)
        
        # feas_j_to_i = [e_j <= l_i - s_i - tau_ij - s_j]
        feas_j_to_i = (node_j.e <= node_i.l - node_i.s - tau_ij - node_j.s)
        
        return feas_i_to_j, feas_j_to_i

    def _compute_slacks_and_order(self, current_graph: Graph, node_i: Node, node_j: Node) -> tuple[str, float]:
        """
        Computes slacks for both orders and selects the order with larger slack.
        Returns (pi_order_string, selected_slack)
        Takes the current_graph to ensure nodes are from the correct graph state.
        """
        tau_ij = compute_euclidean_tau(node_i, node_j) # Use the global helper

        # slack_i_to_j = (l_j - s_j - tau_ij) - (e_i + s_i)
        slack_i_to_j = (node_j.l - node_j.s - tau_ij) - (node_i.e + node_i.s)

        # slack_j_to_i = (l_i - s_i - tau_ij) - (e_j + s_j)
        slack_j_to_i = (node_i.l - node_i.s - tau_ij) - (node_j.e + node_j.s)

        if slack_i_to_j >= slack_j_to_i:
            return f"{node_i.id} -> {node_j.id}", slack_i_to_j
        else:
            return f"{node_j.id} -> {node_i.id}", slack_j_to_i

    def _compute_new_window(self, current_graph: Graph, node_i: Node, node_j: Node, pi_order: str) -> tuple[float, float]:
        """
        Computes the tightened time window [e', l'] for the super-node.
        Takes the current_graph to ensure nodes are from the correct graph state.
        """
        tau_ij = compute_euclidean_tau(node_i, node_j) # Use the global helper
        
        # Parse pi_order to determine which node comes first
        first_node_id, _, second_node_id = pi_order.split(' ')
        
        if first_node_id == node_i.id: # Order is i -> j
            e_prime = max(node_i.e, node_j.e - (node_i.s + tau_ij))
            l_prime = min(node_i.l + node_j.s + tau_ij, node_j.l)
        else: # Order is j -> i
            e_prime = max(node_j.e, node_i.e - (node_j.s + tau_ij))
            l_prime = min(node_j.l + node_i.s + tau_ij, node_i.l)
            
        return e_prime, l_prime

    def _reconnect_neighbors_conservatively(self, current_graph: Graph, super_node: Node, node_i: Node, node_j: Node):
        """
        Reconnects neighbors of node_i and node_j to the new super_node.
        This function is called *after* the super_node has been added to the graph,
        and *before* node_i and node_j are removed.
        
        It adds new edges from super_node to the current neighbors of node_i and node_j.
        It explicitly *does not* remove old edges, as Graph.remove_node will handle that
        when node_i and node_j are finally removed from the graph.
        
        NOTE: "Recomputing tau conservatively" is simplified here by calculating
        Euclidean distance from the super-node's midpoint to the neighbor.
        A truly conservative approach might involve more complex logic
        (e.g., shortest path calculations on original graph, or max travel time),
        especially if neighbors themselves are being merged in the same level.
        """
        # Collect all unique neighbors of i and j, excluding i, j themselves, and the depot
        # These neighbors are still in the graph at this point.
        neighbors_of_i = current_graph.get_neighbors(node_i.id)
        neighbors_of_j = current_graph.get_neighbors(node_j.id)
        
        # Combine neighbors and exclude the merged nodes themselves, and the depot.
        # Also exclude the super_node itself if it somehow got into the neighbor list (safety).
        all_neighbors_ids = (neighbors_of_i.union(neighbors_of_j)) - {node_i.id, node_j.id, super_node.id, self.depot_id}

        # Add new edges from super_node to its unique neighbors
        for neighbor_id in all_neighbors_ids:
            # Check if the neighbor_id still exists in the graph.
            # This is crucial if `M` contains merges that affect each other's neighbors.
            # If `neighbor_id` was already processed and removed in a prior merge in this level,
            # we should skip it for now. A more robust solution would involve connecting
            # to the super-node representing `neighbor_id` if it was merged.
            if neighbor_id in current_graph.nodes:
                neighbor_node = current_graph.nodes[neighbor_id]
                new_tau = compute_euclidean_tau(super_node, neighbor_node) # Use the global helper
                current_graph.add_edge(super_node.id, neighbor_id, new_tau)
            else:
                # This case indicates that the neighbor was already removed (likely merged)
                # in a previous iteration of the 'M' loop in the current level.
                # For a fully robust solution, one would need a mapping from original node IDs
                # to their current super-node representation to ensure connectivity.
                # For now, we skip adding an edge to a non-existent node.
                pass # This is where more advanced reconnection logic would go.


    def coarsen(self) -> tuple[Graph, list]:
        """
        Performs the multilevel graph coarsening.
        Returns the coarsened graph and the list of merge layers.
        """
        G_prime = copy.deepcopy(self.graph) # Start with a deep copy of the original graph
        n_0 = len(G_prime.nodes)

        level = 0
        logger.info(f"--- Starting Coarsening ---")
        logger.info(f"Initial graph size: {len(G_prime.nodes)} nodes, {len(G_prime.edges)} edges")

        while len(G_prime.nodes) > self.P * n_0:
            level += 1
            logger.info(f"\n--- Coarsening Level {level} ---")
            
            # 1. Compute D_ij for all edges in G_prime
            for edge in G_prime.edges:
                edge.D_ij = self._compute_D_ij(G_prime, edge) # Pass G_prime

            # 2. Sort edges by D_ij ascending
            sorted_edges = sorted(G_prime.edges, key=lambda e: e.D_ij)

            # 3. Let rho be the weight at index floor(0.1 * |E| * radiusCoeff)
            num_edges_to_consider = math.floor(0.1 * len(sorted_edges) * self.radiusCoeff)
            rho = sorted_edges[min(num_edges_to_consider, len(sorted_edges) - 1)].D_ij if sorted_edges else 0

            M = [] # List of merges for this iteration: (node_i_id, node_j_id, pi_order, e_prime, l_prime)
            U = {self.depot_id} # Set of used nodes (depot never merges) for this level

            # 4. Identify pairs for merging
            for edge in sorted_edges:
                i_id, j_id = edge.u_id, edge.v_id
                
                # Skip if either node is already used or is the depot
                if i_id in U or j_id in U:
                    continue
                
                node_i = G_prime.nodes[i_id] # Get nodes from the current G_prime
                node_j = G_prime.nodes[j_id] # Get nodes from the current G_prime

                # Only consider edges within the rho threshold
                if edge.D_ij > rho:
                    continue

                # Evaluate feasibility
                feas_i_to_j, feas_j_to_i = self._evaluate_feasibility(G_prime, node_i, node_j) # Pass G_prime

                if not (feas_i_to_j or feas_j_to_i):
                    continue

                # Select order by larger slack
                pi_order, _ = self._compute_slacks_and_order(G_prime, node_i, node_j) # Pass G_prime

                # Compute new time window
                e_prime, l_prime = self._compute_new_window(G_prime, node_i, node_j, pi_order) # Pass G_prime
                
                # Aggregate demand for super-node
                demand_ij = node_i.demand + node_j.demand

                M.append((i_id, j_id, pi_order, e_prime, l_prime, demand_ij))
                U.add(i_id)
                U.add(j_id)
            
            if not M:
                logger.warning(
                    "No feasible merges found at this level. Breaking coarsening loop."
                )
                break # No more merges possible

            logger.info(f"  Found {len(M)} merges at this level.")
            
            # 5. Perform merges and update G_prime
            nodes_to_remove_this_level = set()
            for i_id, j_id, pi_order, e_prime, l_prime, demand_ij in M:
                node_i = G_prime.nodes[i_id]
                node_j = G_prime.nodes[j_id]

                # Create super-node at midpoint
                super_node_id = f"SN_{node_i.id}_{node_j.id}"
                mid_x = (node_i.x + node_j.x) / 2
                mid_y = (node_i.y + node_j.y) / 2
                
                # Aggregate service duration
                s_ij = node_i.s + node_j.s
                
                # Calculate central time for super-node
                t_ij = (e_prime + (l_prime - s_ij)) / 2 if (l_prime - s_ij) >= 0 else e_prime

                # Aggregate original nodes for inflation
                original_nodes_in_super = list(set(node_i.original_nodes + node_j.original_nodes))

                super_node = Node(super_node_id, mid_x, mid_y, s_ij, e_prime, l_prime, demand_ij,
                                  is_super_node=True, original_nodes=original_nodes_in_super)
                
                G_prime.add_node(super_node) # Add super-node to the current G_prime
                
                # Store merge details for inflation
                self.merge_layers.append((super_node.id, node_i.id, node_j.id, pi_order))

                # Reconnect neighbors to the super-node, passing the current G_prime
                self._reconnect_neighbors_conservatively(G_prime, super_node, node_i, node_j)
                
                nodes_to_remove_this_level.add(i_id)
                nodes_to_remove_this_level.add(j_id)
            
            # Remove merged nodes from G_prime
            for node_id in nodes_to_remove_this_level:
                G_prime.remove_node(node_id)
            
            logger.info(f"  Current graph size: {len(G_prime.nodes)} nodes, {len(G_prime.edges)} edges")

        logger.info("\n--- Coarsening Finished ---")
        return G_prime, self.merge_layers

    def inflate_route(self, coarsened_routes: list) -> list:
        """
        Inflates a list of routes from the coarsened graph back to the original graph.
        
        Args:
            coarsened_routes (list): A list of lists of node IDs representing tours
                                     on the coarsened graph.
        Returns:
            list: A list of lists of node IDs representing the inflated tours on the
                  original graph.
        """
        all_inflated_routes = []
        logger.info(f"\n--- Starting Inflation Process ---")
        logger.info(f"Initial coarsened routes: {coarsened_routes}")

        for route_idx, coarsened_route in enumerate(coarsened_routes):
            inflated_route = list(coarsened_route) # Start with a copy of the current coarsened route
            
            if not inflated_route: # Skip empty routes that might be generated by the solver
                continue

            logger.info(f"  Inflating Route {route_idx + 1}: {inflated_route}")

            # Iterate through merge layers in reverse order
            for super_node_id, node_i_id, node_j_id, pi_order in reversed(self.merge_layers):
                # Determine the ordered pair based on pi_order
                first_node_in_pair, _, second_node_in_pair = pi_order.split(' ')
                
                # Create the ordered sub-route for replacement
                ordered_pair = []
                if first_node_in_pair == node_i_id:
                    ordered_pair.append(node_i_id)
                    ordered_pair.append(node_j_id)
                else:
                    ordered_pair.append(node_j_id)
                    ordered_pair.append(node_i_id)

                # Find and replace all occurrences of the super-node in the current inflated_route
                new_inflated_route_segment = []
                replaced_this_layer = False
                for node_in_route in inflated_route:
                    if node_in_route == super_node_id:
                        new_inflated_route_segment.extend(ordered_pair)
                        replaced_this_layer = True
                    else:
                        new_inflated_route_segment.append(node_in_route)
                
                inflated_route = new_inflated_route_segment
                if replaced_this_layer:
                    logger.info(f"    Inflated {super_node_id} to {ordered_pair}. Current Route {route_idx + 1}: {inflated_route}")
                
            all_inflated_routes.append(inflated_route)
            logger.info(f"  Route {route_idx + 1} Inflation Finished. Final: {inflated_route}")
            
        logger.info(f"--- Inflation Finished ---")
        logger.info(f"Final inflated routes: {all_inflated_routes}")
        return all_inflated_routes

# --- 5. GreedySolver Class ---
class GreedySolver:
    """
    A simple greedy heuristic solver for VRPTW.
    Generates multiple routes if a single vehicle cannot serve all customers.
    """
    def __init__(self, graph: Graph, depot_id: str, vehicle_capacity: float):
        self.graph = graph
        self.depot_id = depot_id
        self.vehicle_capacity = vehicle_capacity

    def solve(self) -> tuple[list, dict]:
        """
        Generates routes on the given graph using a simple greedy heuristic.
        Prioritizes visiting the closest feasible unvisited node.
        Dispatches new vehicles as needed.

        Returns:
            tuple: A tuple containing:
                - list: A list of generated routes (each route is a list of node IDs).
                - dict: A dictionary of aggregated metrics for all routes.
        """
        all_routes = []
        
        # All nodes in the graph that are not the depot
        all_customers = {node_id for node_id in self.graph.nodes.keys() if node_id != self.depot_id}
        unvisited_customers = set(all_customers) # Customers remaining to be visited

        logger.info(f"\n--- Starting Greedy Solver on graph with depot {self.depot_id} ---")
        
        vehicle_count = 0
        while unvisited_customers:
            vehicle_count += 1
            current_route = [self.depot_id]
            current_node_id = self.depot_id
            current_time = self.graph.nodes[self.depot_id].e # Each vehicle starts at depot's earliest time window
            current_load = 0.0 # Current load of the vehicle

            logger.info(f"  Dispatching Vehicle {vehicle_count}. Initial state: Current Node={current_node_id}, Current Time={current_time:.2f}, Load={current_load:.2f}")

            vehicle_made_progress_in_this_route = False # Flag to track if the current vehicle adds any customers

            while True:
                best_next_node_id = None
                min_travel_time = float('inf')
                
                current_node = self.graph.nodes[current_node_id]

                # Find the closest feasible unvisited customer for the current vehicle
                feasible_candidates = []
                for candidate_node_id in unvisited_customers:
                    candidate_node = self.graph.nodes[candidate_node_id]

                    # Check capacity
                    if current_load + candidate_node.demand > self.vehicle_capacity:
                        continue  # Cannot add this customer due to capacity

                    travel_time_to_candidate = compute_euclidean_tau(current_node, candidate_node)
                    arrival_time_at_candidate = current_time + travel_time_to_candidate
                    service_start_time_at_candidate = max(arrival_time_at_candidate, candidate_node.e)

                    # Check time window and ensure return to depot is feasible
                    if service_start_time_at_candidate <= candidate_node.l:
                        finish_time_at_candidate = service_start_time_at_candidate + candidate_node.s
                        depot_node = self.graph.nodes[self.depot_id]
                        travel_back_to_depot = compute_euclidean_tau(candidate_node, depot_node)
                        arrival_back_to_depot = finish_time_at_candidate + travel_back_to_depot

                        if arrival_back_to_depot <= depot_node.l:
                            feasible_candidates.append((travel_time_to_candidate, candidate_node_id))
                
                if feasible_candidates:
                    # Sort feasible candidates by travel time to find the closest
                    feasible_candidates.sort(key=lambda x: x[0])
                    min_travel_time, best_next_node_id = feasible_candidates[0]

                if best_next_node_id:
                    next_node = self.graph.nodes[best_next_node_id]
                    
                    travel_time_to_next = compute_euclidean_tau(current_node, next_node)
                    arrival_time_at_next = current_time + travel_time_to_next
                    service_start_time_at_next = max(arrival_time_at_next, next_node.e)
                    
                    current_time = service_start_time_at_next + next_node.s
                    current_load += next_node.demand # Update load
                    
                    current_route.append(best_next_node_id)
                    unvisited_customers.remove(best_next_node_id)
                    current_node_id = best_next_node_id
                    vehicle_made_progress_in_this_route = True # Customer added!
                    logger.info(f"    Vehicle {vehicle_count}: Visited {best_next_node_id}. Current Node={current_node_id}, Current Time={current_time:.2f}, Load={current_load:.2f}")
                else:
                    logger.info(f"    Vehicle {vehicle_count}: No more feasible unvisited customers from {current_node_id}. Ending current route.")
                    break 
            
            # Current vehicle returns to depot
            if current_node_id != self.depot_id:
                depot_node = self.graph.nodes[self.depot_id]
                current_node = self.graph.nodes[current_node_id]
                travel_time_to_depot = compute_euclidean_tau(current_node, depot_node)
                arrival_time_at_depot = current_time + travel_time_to_depot
                
                if arrival_time_at_depot <= depot_node.l:
                    current_route.append(self.depot_id)
                    logger.info(f"    Vehicle {vehicle_count}: Returned to depot. Route: {current_route}")
                else:
                    logger.warning(
                        f"    Vehicle {vehicle_count}: Warning: Cannot return to depot within its time window. Route ends at {current_node_id}. Final time: {arrival_time_at_depot:.2f} (Depot L: {depot_node.l:.2f})"
                    )
                    current_route.append(self.depot_id) # Still append for route completeness, but note infeasibility
            else:
                # If the current_node_id is already the depot (e.g., if no customers were visited by this vehicle)
                if current_route[0] != self.depot_id: # Should not happen with current logic
                    current_route.insert(0, self.depot_id)
                if current_route[-1] != self.depot_id:
                    current_route.append(self.depot_id)

            all_routes.append(current_route)
            
            # Check for infinite loop condition: if a vehicle was dispatched but visited no customers,
            # and there are still unvisited customers, it means we are stuck.
            if not vehicle_made_progress_in_this_route and unvisited_customers:
                logger.warning(
                    f"  Stuck: Vehicle {vehicle_count} dispatched but could not visit any customer. {len(unvisited_customers)} customers remaining. Breaking to prevent infinite loop."
                )
                break # Break the outer loop to prevent infinite vehicle dispatch

            if not unvisited_customers:
                logger.info("  All customers visited.")
                break
            else:
                logger.info(f"  {len(unvisited_customers)} customers remaining. Dispatching new vehicle.")
        
        logger.info(f"--- Greedy Solver Finished ---")
        
        metrics = calculate_route_metrics(self.graph, all_routes, self.depot_id, self.vehicle_capacity)
        if unvisited_customers:
            metrics["is_feasible"] = False
        return all_routes, metrics

# --- 6. SavingsSolver Class (Clarke and Wright) ---
class SavingsSolver:
    """
    Implements the Clarke and Wright Savings Algorithm for VRPTW.
    Generates multiple routes.
    """
    def __init__(self, graph: Graph, depot_id: str, vehicle_capacity: float):
        self.graph = graph
        self.depot_id = depot_id
        self.vehicle_capacity = vehicle_capacity

    def _calculate_savings(self) -> list:
        """
        Calculates savings for combining customer pairs (i, j).
        Saving S_ij = C(depot, i) + C(j, depot) - C(i, j)
        where C is travel time (tau).
        """
        savings = []
        customer_ids = [node_id for node_id in self.graph.nodes if node_id != self.depot_id]
        depot_node = self.graph.nodes[self.depot_id]

        for i in range(len(customer_ids)):
            for j in range(i + 1, len(customer_ids)):
                id_i = customer_ids[i]
                id_j = customer_ids[j]
                node_i = self.graph.nodes[id_i]
                node_j = self.graph.nodes[id_j]

                # Savings are only for combining routes ending at depot and starting from depot
                # S_ij = d(D,i) + d(j,D) - d(i,j)
                # Note: For VRPTW, the time windows and service times make simple distance savings insufficient
                # A more robust savings calculation would consider temporal aspects.
                # For simplicity, we use Euclidean distance as travel time (tau).
                tau_di = compute_euclidean_tau(depot_node, node_i)
                tau_jd = compute_euclidean_tau(node_j, depot_node)
                tau_ij = compute_euclidean_tau(node_i, node_j)
                
                saving = tau_di + tau_jd - tau_ij
                savings.append((saving, id_i, id_j))
        
        # Sort savings in descending order
        savings.sort(key=lambda x: x[0], reverse=True)
        return savings

    def _check_merge_feasibility(self, route1: list, route2: list, merge_point_i: str, merge_point_j: str) -> bool:
        """
        Checks if merging two routes is feasible regarding time windows and capacity.
        Assumes merge_point_i is the end of route1 (before depot) and merge_point_j is start of route2 (after depot).
        
        Args:
            route1 (list): First route (e.g., [D, ..., merge_point_i, D])
            route2 (list): Second route (e.g., [D, merge_point_j, ..., D])
            merge_point_i (str): ID of the last customer in route1 before depot.
            merge_point_j (str): ID of the first customer in route2 after depot.
            
        Returns:
            bool: True if merge is feasible, False otherwise.
        """
        # Form the proposed merged route: [D, ..., route1_nodes_before_i, merge_point_i, merge_point_j, route2_nodes_after_j, ..., D]
        
        # Find index of merge_point_i in route1 (excluding final depot)
        idx_i = -1
        for k in range(len(route1) - 1): # Exclude last depot
            if route1[k] == merge_point_i:
                idx_i = k
                break
        
        # Find index of merge_point_j in route2 (excluding initial depot)
        idx_j = -1
        for k in range(1, len(route2)): # Exclude first depot
            if route2[k] == merge_point_j:
                idx_j = k
                break

        if idx_i == -1 or idx_j == -1: # Should not happen if merge points are valid
            return False

        # Construct the candidate merged route
        # Take route1 up to and including merge_point_i, then route2 from merge_point_j onwards
        candidate_route = route1[:idx_i+1] + route2[idx_j:]
        
        # Check if the combined route starts and ends at depot, if not, fix it
        if candidate_route[0] != self.depot_id:
            candidate_route.insert(0, self.depot_id)
        if candidate_route[-1] != self.depot_id:
            candidate_route.append(self.depot_id)

        # Simulate the route to check feasibility
        current_time = self.graph.nodes[self.depot_id].e
        current_load = 0.0

        for k in range(len(candidate_route) - 1):
            from_node_id = candidate_route[k]
            to_node_id = candidate_route[k+1]

            from_node = self.graph.nodes[from_node_id]
            to_node = self.graph.nodes[to_node_id]

            # Capacity check (only for customer nodes)
            if to_node_id != self.depot_id:
                current_load += to_node.demand
                if current_load > self.vehicle_capacity:
                    return False # Capacity violation

            # Travel time
            travel_time = compute_euclidean_tau(from_node, to_node)
            arrival_time = current_time + travel_time
            
            # Service start time
            service_start_time = max(arrival_time, to_node.e)

            # Time window check (arriving too late)
            if service_start_time > to_node.l:
                return False # Time window violation

            # Update current time
            current_time = service_start_time + to_node.s
        
        # Final check: does the vehicle return to depot within its time window?
        # The last 'current_time' is after serving the last customer.
        # Now, travel back to depot.
        last_customer_node = self.graph.nodes[candidate_route[-2]] # Second to last node is the last customer
        depot_node = self.graph.nodes[self.depot_id]
        travel_time_to_depot = compute_euclidean_tau(last_customer_node, depot_node)
        final_arrival_at_depot = current_time + travel_time_to_depot
        
        if final_arrival_at_depot > depot_node.l:
            return False # Depot time window violation

        return True # Merge is feasible

    def solve(self) -> tuple[list, dict]:
        """
        Generates routes using the Clarke and Wright Savings Algorithm.
        
        Returns:
            tuple: A tuple containing:
                - list: A list of generated routes (each route is a list of node IDs).
                - dict: A dictionary of aggregated metrics for all routes.
        """
        logger.info(f"\n--- Starting Savings Solver on graph with depot {self.depot_id} ---")

        # Step 1: Initialize routes (each customer gets its own route)
        routes = {} # {customer_id: [depot_id, customer_id, depot_id]}
        customer_ids = [node_id for node_id in self.graph.nodes if node_id != self.depot_id]
        for cust_id in customer_ids:
            routes[cust_id] = [self.depot_id, cust_id, self.depot_id]
        
        # Map customer ID to the route it belongs to (for quick lookup)
        customer_to_route_map = {cust_id: cust_id for cust_id in customer_ids}

        logger.info(f"  Initial routes: {len(customer_ids)} individual routes.")

        # Step 2: Calculate savings
        savings = self._calculate_savings()
        logger.info(f"  Calculated {len(savings)} potential savings.")

        # Step 3: Iterate through savings and attempt merges
        merged_any_this_iteration = True
        while merged_any_this_iteration:
            merged_any_this_iteration = False
            for saving_value, id_i, id_j in savings:
                # Ensure i and j are still customers and not yet merged into the same route
                if id_i not in customer_to_route_map or id_j not in customer_to_route_map:
                    continue # One or both nodes have already been merged away

                route_id_i = customer_to_route_map[id_i]
                route_id_j = customer_to_route_map[id_j]

                if route_id_i == route_id_j:
                    continue # Already in the same route

                route_i = routes[route_id_i]
                route_j = routes[route_id_j]

                # For Savings algorithm, we merge routes like:
                # Route A: [D, ..., X, i, D]
                # Route B: [D, j, Y, ..., D]
                # Merged: [D, ..., X, i, j, Y, ..., D]
                # So, i must be the last customer in route_i (before its depot)
                # and j must be the first customer in route_j (after its depot).

                # Check if i is the last customer in route_i (before final depot)
                is_i_endpoint = (route_i[-2] == id_i)
                # Check if j is the first customer in route_j (after initial depot)
                is_j_endpoint = (route_j[1] == id_j)

                # Check if i is the last customer in route_j (before final depot)
                is_i_endpoint_j = (route_j[-2] == id_i)
                # Check if j is the first customer in route_i (after initial depot)
                is_j_endpoint_i = (route_i[1] == id_j)

                # Case 1: Merge Route_i's end with Route_j's start (i -> j)
                if is_i_endpoint and is_j_endpoint:
                    # Check feasibility for merging route_i[:-1] + route_j[1:]
                    if self._check_merge_feasibility(route_i, route_j, id_i, id_j):
                        proposed_merged_route = route_i[:-1] + route_j[1:]
                        new_route_id = f"R_{route_id_i}_{route_id_j}"
                        
                        # Update maps and routes
                        for customer_in_old_route in route_i[1:-1]:
                            customer_to_route_map[customer_in_old_route] = new_route_id
                        for customer_in_old_route in route_j[1:-1]:
                            customer_to_route_map[customer_in_old_route] = new_route_id
                        
                        routes[new_route_id] = proposed_merged_route
                        del routes[route_id_i]
                        del routes[route_id_j]
                        merged_any_this_iteration = True
                        logger.info(f"  Merged routes {route_id_i} and {route_id_j} via ({id_i} -> {id_j}) with saving {saving_value:.2f}. New route: {new_route_id}")
                        break # Restart iteration over savings after a merge
                
                # Case 2: Merge Route_j's end with Route_i's start (j -> i)
                # This is symmetric, so we swap i and j roles
                elif is_j_endpoint_i and is_i_endpoint_j: # This condition is for j at end of route_i and i at start of route_j (reversed logic)
                    # Correct logic for j->i merge:
                    # Route A: [D, ..., X, j, D]
                    # Route B: [D, i, Y, ..., D]
                    # Merged: [D, ..., X, j, i, Y, ..., D]
                    # So, j must be the last customer in route_j (before depot)
                    # and i must be the first customer in route_i (after depot).
                    if route_j[-2] == id_j and route_i[1] == id_i: # Check original roles
                        if self._check_merge_feasibility(route_j, route_i, id_j, id_i): # Pass route_j as first route, route_i as second
                            proposed_merged_route = route_j[:-1] + route_i[1:]
                            new_route_id = f"R_{route_id_j}_{route_id_i}" # Naming convention for the new route
                            
                            # Update maps and routes
                            for customer_in_old_route in route_i[1:-1]:
                                customer_to_route_map[customer_in_old_route] = new_route_id
                            for customer_in_old_route in route_j[1:-1]:
                                customer_to_route_map[customer_in_old_route] = new_route_id
                            
                            routes[new_route_id] = proposed_merged_route
                            del routes[route_id_i]
                            del routes[route_id_j]
                            merged_any_this_iteration = True
                            logger.info(f"  Merged routes {route_id_j} and {route_id_i} via ({id_j} -> {id_i}) with saving {saving_value:.2f}. New route: {new_route_id}")
                            break # Restart iteration over savings after a merge

            # If no merges were made in an entire pass, stop.
            if not merged_any_this_iteration:
                break

        final_routes_list = list(routes.values())
        logger.info(f"--- Savings Solver Finished. Found {len(final_routes_list)} routes. ---")
        
        metrics = calculate_route_metrics(self.graph, final_routes_list, self.depot_id, self.vehicle_capacity)
        return final_routes_list, metrics


# --- 7. CSV Loading Function ---
def load_graph_from_csv(file_path: str) -> tuple[Graph, str, float]:
    """Load a Solomon VRPTW instance from a CSV file.

    The CSVs shipped with this repository have a single header line followed by
    the data rows.  Previous logic expected several header lines and attempted to
    parse the vehicle capacity from one of them, which resulted in skipping the
    first eight customers and mis-detecting the capacity.  The revised version
    simply reads the file from the start and uses Solomon's standard capacity of
    ``200``.

    Args:
        file_path: Path to the CSV file.

    Returns:
        ``(graph, depot_id, vehicle_capacity)`` with ``depot_id`` taken from the
        first row in the file.
    """
    graph = Graph()
    depot_id = None
    vehicle_capacity = 200.0

    # Define the actual column headers found in Solomon datasets
    solomon_headers = [
        'CUST NO.', 'XCOORD.', 'YCOORD.', 'DEMAND', 'READY TIME', 'DUE DATE', 'SERVICE TIME'
    ]

    try:
        with open(file_path, newline="") as f:
            reader = csv.DictReader(f)

            for i, row in enumerate(reader):
                if not row:
                    continue

                try:
                    node_id = row[solomon_headers[0]].strip()
                    x = parse_float(row[solomon_headers[1]])
                    y = parse_float(row[solomon_headers[2]])
                    demand = parse_float(row[solomon_headers[3]])
                    e = parse_float(row[solomon_headers[4]])
                    l = parse_float(row[solomon_headers[5]])
                    s = parse_float(row[solomon_headers[6]])

                    node = Node(node_id, x, y, s, e, l, demand)
                    graph.add_node(node)

                    if i == 0:
                        depot_id = node_id
                except (ValueError, KeyError) as data_error:
                    raise ValueError(
                        f"Error processing data in row {i+1} of {file_path}. Row content: {row}. Details: {data_error}"
                    ) from data_error

                
        if depot_id is None:
            raise ValueError("No nodes found in CSV data or depot not identified.")

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
        # Re-raise the original exception for the calling code to handle
        raise
    except ValueError as e:
        logger.error(f"Error processing CSV data: {e}")
        # Re-raise the original exception for the calling code to handle
        raise
    except Exception as e:
        logger.exception(f"An unexpected error occurred while loading CSV: {e}")
        import traceback
        traceback.print_exc() # Print full traceback for unexpected errors
        # Re-raise the original exception for the calling code to handle
        raise

# --- Example Usage ---
if __name__ == "__main__":
    # Define the base directory where 'solomon_dataset' is located
    # Assuming the Python script is in the same directory as 'solomon_dataset'
    base_dataset_dir = 'solomon_dataset' 

    # List to store full paths of all CSV files found
    all_csv_file_paths = []

    # Walk through the base_dataset_dir to find all CSV files
    for root, dirs, files in os.walk(base_dataset_dir):
        for file in files:
            if file.endswith('.csv'):
                full_path = os.path.join(root, file)
                all_csv_file_paths.append(full_path)
    
    # Sort the file paths for consistent processing order (e.g., C101, C102, ..., R101, etc.)
    all_csv_file_paths.sort()

    # Store results for all files
    all_results = {}

    for csv_file_path in all_csv_file_paths:
        # Extract just the file name for display in results
        file_name_only = os.path.basename(csv_file_path)

        logger.info(f"\n\n=====================================================================================")
        logger.info(f"Processing file: {csv_file_path}")
        logger.info(f"=====================================================================================")

        # --- Load graph from CSV ---
        try:
            initial_graph, depot_id, VEHICLE_CAPACITY = load_graph_from_csv(csv_file_path)
        except Exception as e:
            logger.warning(f"Skipping {csv_file_path} due to error loading graph: {e}")
            continue # Skip to the next file if loading fails

        current_file_results = {}

        logger.info("\n--- Initial Graph Nodes (from CSV) ---")
        for node_id, node in list(initial_graph.nodes.items())[:5]: # Print first 5 nodes
            logger.info(node)
        logger.info(f"... and {len(initial_graph.nodes) - 5} more nodes.")
        logger.info("\n--- Initial Graph Edges (from CSV, first 5) ---")
        for i, edge in enumerate(initial_graph.edges):
            if i >= 5: break
            logger.info(edge)
        logger.info(f"Total initial edges: {len(initial_graph.edges)}")

        # --- Solve directly on the uncoarsened graph with Greedy Solver ---
        logger.info("\n\n=== Solving on UNCOARSENED Graph (Greedy Solver) ===")
        uncoarsened_greedy_solver = GreedySolver(initial_graph, depot_id, VEHICLE_CAPACITY)
        uncoarsened_greedy_routes, uncoarsened_greedy_metrics = uncoarsened_greedy_solver.solve()
        current_file_results['Uncoarsened Greedy'] = uncoarsened_greedy_metrics
        
        logger.info("\n--- Metrics for UNCOARSENED Greedy Routes ---")
        logger.info(f"Routes: {uncoarsened_greedy_routes}")
        for key, value in uncoarsened_greedy_metrics.items():
            if isinstance(value, float):
                logger.info(f"{key.replace('_', ' ').title()}: {value:.2f}")
            else:
                logger.info(f"{key.replace('_', ' ').title()}: {value}")


        # --- Solve directly on the uncoarsened graph with Savings Solver ---
        logger.info("\n\n=== Solving on UNCOARSENED Graph (Savings Solver) ===")
        uncoarsened_savings_solver = SavingsSolver(initial_graph, depot_id, VEHICLE_CAPACITY)
        uncoarsened_savings_routes, uncoarsened_savings_metrics = uncoarsened_savings_solver.solve()
        current_file_results['Uncoarsened Savings'] = uncoarsened_savings_metrics
        
        logger.info("\n--- Metrics for UNCOARSENED Savings Routes ---")
        logger.info(f"Routes: {uncoarsened_savings_routes}")
        for key, value in uncoarsened_savings_metrics.items():
            if isinstance(value, float):
                logger.info(f"{key.replace('_', ' ').title()}: {value:.2f}")
            else:
                logger.info(f"{key.replace('_', ' ').title()}: {value}")


        # --- Coarsening Process ---
        logger.info("\n\n=== Starting Coarsening Process ===")
        coarsener = SpatioTemporalGraphCoarsener(
            graph=initial_graph,
            alpha=0.5,
            beta=0.5,
            P=0.5, # Reduce to 50% of original nodes
            radiusCoeff=1.0,
            depot_id=depot_id
        )
        coarsened_graph, merge_layers = coarsener.coarsen()

        logger.info("\n--- Final Coarsened Graph Nodes ---")
        for node_id, node in list(coarsened_graph.nodes.items())[:5]: # Print first 5 nodes
            logger.info(node)
        logger.info(f"... and {len(coarsened_graph.nodes) - 5} more nodes.")
        logger.info("\n--- Final Coarsened Graph Edges (first 5) ---")
        for i, edge in enumerate(coarsened_graph.edges):
            if i >= 5: break
            logger.info(edge)
        logger.info(f"Total final edges: {len(coarsened_graph.edges)}")

        logger.info("\n--- Merge Layers (for Inflation) ---")
        for layer in merge_layers[:5]: # Print first 5 merge layers
            super_node_id, node_i_id, node_j_id, pi_order = layer
            logger.info(f"Super-node: {super_node_id} formed from {node_i_id} and {node_j_id} in order {pi_order}")
        logger.info(f"... and {len(merge_layers) - 5} more merge layers.")

        logger.info("\n--- Original nodes represented by final super-nodes (first 5) ---")
        count_super_nodes_printed = 0
        for node_id, node in coarsened_graph.nodes.items():
            if node.is_super_node:
                logger.info(f"Super-node {node_id} represents original nodes: {node.original_nodes}")
                count_super_nodes_printed += 1
                if count_super_nodes_printed >= 5:
                    break
        logger.info(f"... and more super-nodes.")


        # --- Solve on the coarsened graph with Greedy Solver and then inflate ---
        logger.info("\n\n=== Solving on COARSENED Graph (Greedy Solver) and Inflating ===")
        coarsened_greedy_solver = GreedySolver(coarsened_graph, depot_id, VEHICLE_CAPACITY)
        coarsened_greedy_routes, coarsened_greedy_metrics = coarsened_greedy_solver.solve()
        
        final_inflated_greedy_routes = coarsener.inflate_route(coarsened_greedy_routes)

        # --- Calculate metrics for the inflated Greedy route ---
        inflated_greedy_metrics = calculate_route_metrics(initial_graph, final_inflated_greedy_routes, depot_id, VEHICLE_CAPACITY)
        current_file_results['Inflated Greedy'] = inflated_greedy_metrics

        logger.info("\n--- Metrics for INFLATED Greedy Routes (on Original Graph) ---")
        logger.info(f"Routes: {final_inflated_greedy_routes}")
        for key, value in inflated_greedy_metrics.items():
            if isinstance(value, float):
                logger.info(f"{key.replace('_', ' ').title()}: {value:.2f}")
            else:
                logger.info(f"{key.replace('_', ' ').title()}: {value}")

        # --- Solve on the coarsened graph with Savings Solver and then inflate ---
        logger.info("\n\n=== Solving on COARSENED Graph (Savings Solver) and Inflating ===")
        coarsened_savings_solver = SavingsSolver(coarsened_graph, depot_id, VEHICLE_CAPACITY)
        coarsened_savings_routes, coarsened_savings_metrics = coarsened_savings_solver.solve()
        
        final_inflated_savings_routes = coarsener.inflate_route(coarsened_savings_routes)

        # --- Calculate metrics for the inflated Savings route ---
        inflated_savings_metrics = calculate_route_metrics(initial_graph, final_inflated_savings_routes, depot_id, VEHICLE_CAPACITY)
        current_file_results['Inflated Savings'] = inflated_savings_metrics

        logger.info("\n--- Metrics for INFLATED Savings Routes (on Original Graph) ---")
        logger.info(f"Routes: {final_inflated_savings_routes}")
        for key, value in inflated_savings_metrics.items():
            if isinstance(value, float):
                logger.info(f"{key.replace('_', ' ').title()}: {value:.2f}")
            else:
                logger.info(f"{key.replace('_', ' ').title()}: {value}")

        # Store results for the current file
        all_results[file_name_only] = current_file_results # Store by just the file name

    # --- Final Summary Comparison Across All Files ---
    logger.info("\n\n=====================================================================================")
    logger.info("======================== FINAL SUMMARY ACROSS ALL FILES =============================")
    logger.info("=====================================================================================")

    metrics_to_compare = [
        "total_distance",
        "total_service_time",
        "total_waiting_time",
        "total_route_duration",
        "total_demand_served",
        "time_window_violations",
        "capacity_violations",
        "num_vehicles",
        "is_feasible"
    ]

    # Print headers for the summary table
    logger.info(f"{'Metric':<25} | {'Uncoarsened Greedy':<20} | {'Uncoarsened Savings':<20} | {'Inflated Greedy':<20} | {'Inflated Savings':<20}")
    logger.info("-" * 120)

    for file_name in sorted(all_results.keys()): # Iterate in sorted order of file names
        results = all_results[file_name]
        logger.info(f"\n--- Results for {file_name} ---")
        
        for metric in metrics_to_compare:
            uncoarsened_greedy_val = results.get('Uncoarsened Greedy', {}).get(metric, 'N/A')
            uncoarsened_savings_val = results.get('Uncoarsened Savings', {}).get(metric, 'N/A')
            inflated_greedy_val = results.get('Inflated Greedy', {}).get(metric, 'N/A')
            inflated_savings_val = results.get('Inflated Savings', {}).get(metric, 'N/A')

            def format_val(val):
                if isinstance(val, float):
                    return f"{val:.2f}"
                return str(val)

            logger.info(f"{metric.replace('_', ' ').title():<25} | {format_val(uncoarsened_greedy_val):<20} | {format_val(uncoarsened_savings_val):<20} | {format_val(inflated_greedy_val):<20} | {format_val(inflated_savings_val):<20}")

    logger.info("\nNote: 'Is Feasible' indicates if any time window or capacity violations were found across all routes.")
    logger.info("A well-functioning coarsening/inflation should ideally result in feasible inflated routes serving all demand.")
