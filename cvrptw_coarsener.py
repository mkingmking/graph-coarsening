import math
import logging

from graph import Graph, compute_euclidean_tau
from node import Node

logger = logging.getLogger(__name__)

class CVRPTWCoarsener:
    """
    Implements a multilevel coarsening algorithm for CVRPTW that includes
    demand penalties to prevent infeasible MetaNodes from being created.
    """
    def __init__(self, graph: Graph, depot_id: str, vehicle_capacity: float,
                 wc: float = 0.4, wd: float = 0.2, wt: float = 0.4):
        """
        Initializes the CVRPTWCoarsener with problem parameters and penalty weights.

        Args:
            graph (Graph): The initial graph with customer and depot nodes.
            depot_id (str): The ID of the depot node.
            vehicle_capacity (float): The maximum capacity of a vehicle.
            wc (float): Weight for the cost penalty. Defaults to 0.4.
            wd (float): Weight for the demand penalty. Defaults to 0.2.
            wt (float): Weight for the time penalty. Defaults to 0.4.
        """
        self.initial_graph = graph
        self.depot_id = depot_id
        self.vehicle_capacity = vehicle_capacity
        self.wc = wc
        self.wd = wd
        self.wt = wt
        self.merge_layers = []  # To store the history of merges
        self.t_max = self._calculate_max_distance()
        logger.info(f"Initialized CVRPTWCoarsener with Q={self.vehicle_capacity}, wc={wc}, wd={wd}, wt={wt}")
        logger.info(f"Max distance for normalization: {self.t_max:.2f}")

    def _calculate_max_distance(self):
        """
        Calculates the maximum possible Euclidean distance between any two nodes
        in the initial graph for normalization purposes.
        """
        max_dist = 0.0
        node_ids = list(self.initial_graph.nodes.keys())
        for i in range(len(node_ids)):
            for j in range(i + 1, len(node_ids)):
                node1 = self.initial_graph.nodes[node_ids[i]]
                node2 = self.initial_graph.nodes[node_ids[j]]
                max_dist = max(max_dist, compute_euclidean_tau(node1, node2))
        return max_dist if max_dist > 0 else 1.0

    def _calculate_penalties(self, node_i: Node, node_j: Node):
        """
        Calculates the three penalties for a potential merge of two nodes.

        Returns:
            A tuple (P_c, P_d, P_t) of penalties, or (inf, inf, inf) if not feasible.
        """
        # --- 1. Cost Penalty (P_c) ---
        tau_ij = compute_euclidean_tau(node_i, node_j)
        P_c = tau_ij / self.t_max

        # --- 2. Demand Penalty (P_d) ---
        combined_demand = node_i.demand + node_j.demand
        if combined_demand > self.vehicle_capacity:
            # This merge is capacity-infeasible, return infinite penalty
            logger.debug(f"Merge of {node_i.id} and {node_j.id} is infeasible (demand violation).")
            return math.inf, math.inf, math.inf

        P_d = combined_demand / self.vehicle_capacity

        # --- 3. Time Penalty (P_t) ---
        # This is a simplified feasibility check.
        # Check A -> B
        t_service_i = node_i.s 
        earliest_arrival_j = node_i.e + t_service_i + tau_ij
        
        feasible_i_to_j = (earliest_arrival_j <= node_j.l)
        
        # Check B -> A
        tau_ji = compute_euclidean_tau(node_j, node_i)
        t_service_j = node_j.s
        earliest_arrival_i = node_j.e + t_service_j + tau_ji
        
        feasible_j_to_i = (earliest_arrival_i <= node_i.l)
        
        if not feasible_i_to_j and not feasible_j_to_i:
            logger.debug(f"Merge of {node_i.id} and {node_j.id} is infeasible (time window violation).")
            return math.inf, math.inf, math.inf

        # For the penalty, we use the slack from both nodes. More slack means it's an easier merge.
        # A higher penalty for a more restrictive combined time window.
        time_slack_i = node_i.l - node_i.e
        time_slack_j = node_j.l - node_j.e
        combined_time_window_length = min(node_i.l, node_j.l) - max(node_i.e, node_j.e)
        
        # Simple penalty based on normalized time window length. Shorter window -> higher penalty.
        max_time_window = 1440 # Assuming a 24h period, adjust as needed.
        P_t = 1 - (combined_time_window_length / max_time_window)
        if P_t < 0: P_t = 0 # Avoid negative penalties

        return P_c, P_d, P_t

    def _calculate_score(self, node_i: Node, node_j: Node):
        """Calculates the weighted score for merging two nodes."""
        P_c, P_d, P_t = self._calculate_penalties(node_i, node_j)
        
        if P_c == math.inf:  # Infeasible merge
            return math.inf
        
        score = self.wc * P_c + self.wd * P_d + self.wt * P_t
        return score

    def _merge_nodes(self, node_a: Node, node_b: Node):
        """
        Merges two nodes into a new MetaNode.
        """
        new_node_id = f"meta_{node_a.id}_{node_b.id}"
        
        # Aggregate attributes
        total_demand = node_a.demand + node_b.demand
        
        # Weighted average for coordinates based on demand
        x_new = (node_a.x * node_a.demand + node_b.x * node_b.demand) / total_demand if total_demand > 0 else (node_a.x + node_b.x) / 2
        y_new = (node_a.y * node_a.demand + node_b.y * node_b.demand) / total_demand if total_demand > 0 else (node_a.y + node_b.y) / 2
        
        # Combine service times and time windows
        s_new = node_a.s + node_b.s
        e_new = min(node_a.e, node_b.e)
        l_new = max(node_a.l, node_b.l)
        
        # The internal sequence of original customers. For simplicity, we just combine them.
        internal_sequence = node_a.original_nodes + node_b.original_nodes
        
        return Node(new_node_id, x_new, y_new, s_new, e_new, l_new, total_demand, is_super_node=True, original_nodes=internal_sequence)

    def coarsen(self, current_graph: Graph):
        """
        Performs the coarsening process on a given graph until only the depot and
        a single MetaNode remain.

        Args:
            current_graph (Graph): The graph to coarsen. This graph will be modified.

        Returns:
            Graph: The final coarsened graph.
            list: The history of merge layers.
        """
        num_customers = len(current_graph.nodes) - 1 # Excluding the depot
        
        while num_customers > 1:
            best_score = math.inf
            best_pair = None
            pi_order = None # Optimal internal sequence order

            customer_nodes = [node for node in current_graph.nodes.values() if node.id != self.depot_id]
            
            # Find the best pair to merge
            for i in range(len(customer_nodes)):
                for j in range(i + 1, len(customer_nodes)):
                    node_i = customer_nodes[i]
                    node_j = customer_nodes[j]
                    
                    score = self._calculate_score(node_i, node_j)
                    
                    if score < best_score:
                        best_score = score
                        best_pair = (node_i.id, node_j.id)
                        # The internal sequence finding is not implemented here.
                        # We'll just assume an order for now.
                        pi_order = node_i.original_nodes + node_j.original_nodes

            if best_pair is None or best_score == math.inf:
                logger.info("No feasible merges remaining. Stopping coarsening.")
                break

            # Perform the merge and update the graph
            node_i_id, node_j_id = best_pair
            node_i = current_graph.nodes[node_i_id]
            node_j = current_graph.nodes[node_j_id]
            
            new_node = self._merge_nodes(node_i, node_j)
            
            # Add the new MetaNode to the graph
            current_graph.add_node(new_node)
            
            # Remove the original nodes from the graph
            current_graph.remove_node(node_i_id)
            current_graph.remove_node(node_j_id)
            
            # Remove old edges and add new edges for the merged node
            # For simplicity in this example, we just remove and then assume a complete graph.
            # In a full implementation, you would recompute edges from the new node to all others.
            
            self.merge_layers.append({
                "super_node_id": new_node.id,
                "original_node_i_id": node_i_id,
                "original_node_j_id": node_j_id,
                "pi_order": pi_order
            })
            
            num_customers = len(current_graph.nodes) - 1
            logger.info(f"Merged nodes {node_i_id} and {node_j_id} into {new_node.id}. Remaining customer nodes: {num_customers}")
            
        logger.info("Coarsening complete.")
        return current_graph, self.merge_layers


    def inflate_route(self, coarse_routes: list) -> list:
        """
        Inflates a solution from the coarsened graph back to the original graph,
        handling multiple layers of coarsening.
        
        Args:
            coarse_routes (list): A list of routes from the coarsened graph,
                                  where each route is a list of node IDs.
                                  
        Returns:
            list: A list of the fully inflated routes on the original graph.
        """
        logger.info(f"--- Starting Inflation ---")
        final_inflated_routes = []
        for route_idx, coarse_route in enumerate(coarse_routes):
            current_route = coarse_route
            
            # Repeatedly expand meta-nodes until no more exist in the route
            while any(node_id.startswith("meta_") for node_id in current_route):
                new_route = []
                for node_id in current_route:
                    if node_id.startswith("meta_"):
                        # Find the internal sequence for this meta-node
                        pi_order = None
                        for layer in self.merge_layers:
                            if layer["super_node_id"] == node_id:
                                pi_order = layer["pi_order"]
                                break
                        if pi_order:
                            new_route.extend(pi_order)
                        else:
                            logger.warning(f"Warning: MetaNode {node_id} not found in merge layers. Skipping.")
                            new_route.append(node_id) # Append it as a placeholder to avoid breaking the route
                    else:
                        new_route.append(node_id)
                current_route = new_route
            final_inflated_routes.append(current_route)
        
        logger.info(f"--- Inflation Finished ---")
        return final_inflated_routes
