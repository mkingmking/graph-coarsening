import math
import copy
import logging

from graph import Graph, compute_euclidean_tau
from node import Node

logger = logging.getLogger(__name__)

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

    def _compute_D_ij(self, current_graph: Graph, edge) -> float:
        """
        Computes the spatio-temporal distance metric D_ij for an edge.
        D_ij = alpha * tau_ij + beta * max(0, e_j - (t_i + s_i + tau_ij))
        Takes the current_graph to ensure nodes are from the correct graph state.
        """
        node_i = current_graph.nodes[edge.u_id]
        node_j = current_graph.nodes[edge.v_id]

        tau_ij = compute_euclidean_tau(node_i, node_j) # Use the global helper

        # Temporal slack calculation (for i -> j order)
        # This is max(0, earliest_service_start_at_j - earliest_arrival_at_j_from_i_considering_i_central_time)
        temporal_slack = max(0, node_j.e - (node_i.t + node_i.s + tau_ij))
        
        return self.alpha * tau_ij + self.beta * temporal_slack

    def _evaluate_feasibility(self, current_graph: Graph, node_i: Node, node_j: Node) -> tuple[bool, bool]:
        """
        Evaluates merge feasibility for i -> j and j -> i, based on service START time windows.
        Returns (feas_i_to_j, feas_j_to_i)
        Takes the current_graph to ensure nodes are from the correct graph state.
        
        Feasibility (i -> j): Earliest service start at j (if coming from i) must be <= l_j.
        Earliest service start at i: node_i.e
        Earliest service finish at i: node_i.e + node_i.s
        Earliest arrival at j: node_i.e + node_i.s + tau_ij
        Earliest service start at j: max(earliest_arrival_at_j, node_j.e)
        Condition: max(node_i.e + node_i.s + tau_ij, node_j.e) <= node_j.l
        Since node_j.e <= node_j.l is assumed, the critical part is:
        node_i.e + node_i.s + tau_ij <= node_j.l
        """
        tau_ij = compute_euclidean_tau(node_i, node_j) # Use the global helper

        # Corrected feasibility for i -> j: earliest start at j must be <= l_j
        feas_i_to_j = (node_i.e + node_i.s + tau_ij <= node_j.l)
        
        # Corrected feasibility for j -> i: earliest start at i must be <= l_i
        feas_j_to_i = (node_j.e + node_j.s + tau_ij <= node_i.l)
        
        return feas_i_to_j, feas_j_to_i

    def _compute_slacks_and_order(self, current_graph: Graph, node_i: Node, node_j: Node) -> tuple[str, float]:
        """
        Computes slacks for both orders and selects the order with larger slack,
        based on service START time windows.
        Returns (pi_order_string, selected_slack)
        Takes the current_graph to ensure nodes are from the correct graph state.
        
        Slack (i -> j): Latest possible service start at j (l_j) - Earliest possible service start at j (if coming from i)
        Earliest possible service start at j (from i): node_i.e + node_i.s + tau_ij
        """
        tau_ij = compute_euclidean_tau(node_i, node_j) # Use the global helper

        # Corrected slack for i -> j
        slack_i_to_j = node_j.l - (node_i.e + node_i.s + tau_ij)

        # Corrected slack for j -> i
        slack_j_to_i = node_i.l - (node_j.e + node_j.s + tau_ij)

        if slack_i_to_j >= slack_j_to_i:
            return f"{node_i.id} -> {node_j.id}", slack_i_to_j
        else:
            return f"{node_j.id} -> {node_i.id}", slack_j_to_i

    def _compute_new_window(self, current_graph: Graph, node_i: Node, node_j: Node, pi_order: str) -> tuple[float, float]:
        """
        Computes the tightened time window [e', l'] for the super-node,
        based on service START time windows.
        Takes the current_graph to ensure nodes are from the correct graph state.
        
        For a super-node SN representing sequence A -> B:
        e'_SN = max(e_A, e_B - (s_A + tau_AB))
        l'_SN = min(l_A, l_B - (s_A + tau_AB))
        """
        tau_ij = compute_euclidean_tau(node_i, node_j) # Use the global helper
        
        # Parse pi_order to determine which node comes first
        first_node_id, _, second_node_id = pi_order.split(' ')
        
        if first_node_id == node_i.id: # Order is i -> j
            # e_prime: Super-node must not start before i.e. Also, if SN starts at S_SN,
            # i finishes at S_SN + s_i, j arrives at S_SN + s_i + tau_ij.
            # j must start by j.e, so S_SN + s_i + tau_ij >= j.e => S_SN >= j.e - s_i - tau_ij
            e_prime = max(node_i.e, node_j.e - (node_i.s + tau_ij))
            
            # l_prime: Super-node must start by i.l. Also, if SN starts at S_SN,
            # i finishes at S_SN + s_i, j arrives at S_SN + s_i + tau_ij.
            # j must start by j.l, so S_SN + s_i + tau_ij <= j.l => S_SN <= j.l - s_i - tau_ij
            l_prime = min(node_i.l, node_j.l - node_i.s - tau_ij)
        else: # Order is j -> i
            # e_prime: Super-node must not start before j.e. Also, if SN starts at S_SN,
            # j finishes at S_SN + s_j, i arrives at S_SN + s_j + tau_ji.
            # i must start by i.e, so S_SN + s_j + tau_ji >= i.e => S_SN >= i.e - s_j - tau_ji
            e_prime = max(node_j.e, node_i.e - (node_j.s + tau_ij)) # Note: tau_ji is same as tau_ij
            
            # l_prime: Super-node must start by j.l. Also, if SN starts at S_SN,
            # j finishes at S_SN + s_j, i arrives at S_SN + s_j + tau_ji.
            # i must start by i.l, so S_SN + s_j + tau_ji <= i.l => S_SN <= i.l - s_j - tau_ji
            l_prime = min(node_j.l, node_i.l - node_j.s - tau_ij)
            
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

