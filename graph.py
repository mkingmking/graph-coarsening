import math
from . node import Node
from .edge import Edge

def compute_euclidean_tau(node1: Node, node2: Node) -> float:
    """
    Computes the Euclidean travel time (distance) between two nodes.
    """
    return math.sqrt((node1.x - node2.x)**2 + (node1.y - node2.y)**2)

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

