from __future__ import annotations

import math
from .node import Node
from .edge import Edge


def compute_euclidean_tau(node1: Node, node2: Node) -> float:
    """Computes the Euclidean travel time (distance) between two nodes."""
    return math.sqrt((node1.x - node2.x)**2 + (node1.y - node2.y)**2)


class Graph:
    """
    Represents the graph with nodes and edges.
    Attributes:
        nodes (dict): Dictionary mapping node ID to Node object.
        edges (list): List of Edge objects.
        adj (dict): Adjacency list mapping node ID to a set of connected node IDs.
    """

    def __init__(self) -> None:
        self.nodes: dict[str, Node] = {}
        self.edges: list[Edge] = []
        self.adj: dict[str, set[str]] = {}

    def add_node(self, node: Node) -> None:
        """Adds a node to the graph."""
        self.nodes[node.id] = node
        if node.id not in self.adj:
            self.adj[node.id] = set()

    def add_edge(self, u_id: str, v_id: str, tau: float) -> None:
        """Adds an edge to the graph, connecting two existing nodes."""
        if u_id not in self.nodes or v_id not in self.nodes:
            raise ValueError(f"Nodes {u_id} or {v_id} not found in graph.")

        # Check if edge already exists to avoid duplicates
        for edge in self.edges:
            if (edge.u_id == u_id and edge.v_id == v_id) or \
               (edge.u_id == v_id and edge.v_id == u_id):
                return  # Edge already exists

        edge = Edge(u_id, v_id, tau)
        self.edges.append(edge)
        self.adj[u_id].add(v_id)
        self.adj[v_id].add(u_id)  # Assuming undirected graph for VRP connections

    def remove_node(self, node_id: str) -> None:
        """Removes a node and all its incident edges from the graph."""
        if node_id not in self.nodes:
            return

        # Remove edges connected to this node
        self.edges = [edge for edge in self.edges if edge.u_id != node_id and edge.v_id != node_id]

        # Remove from adjacency list
        if node_id in self.adj:
            for neighbor_id in list(self.adj[node_id]):  # Iterate over a copy
                if neighbor_id in self.adj:
                    self.adj[neighbor_id].discard(node_id)
            del self.adj[node_id]

        del self.nodes[node_id]

    def get_edge_by_nodes(self, u_id: str, v_id: str) -> Edge | None:
        """Returns an edge object given its two node IDs, or None if not found."""
        for edge in self.edges:
            if (edge.u_id == u_id and edge.v_id == v_id) or \
               (edge.u_id == v_id and edge.v_id == u_id):
                return edge
        return None

    def get_neighbors(self, node_id: str) -> set[str]:
        """Returns a set of neighbor IDs for a given node."""
        return self.adj.get(node_id, set())

    def get_all_edges_for_node(self, node_id: str) -> list[Edge]:
        """Returns a list of edge objects connected to a given node."""
        return [edge for edge in self.edges if edge.u_id == node_id or edge.v_id == node_id]

    def distance(self, u_id: str, v_id: str) -> float:
        """Return the Euclidean travel time between two nodes already in the graph."""
        return compute_euclidean_tau(self.nodes[u_id], self.nodes[v_id])

    def build_complete_edges(self) -> None:
        """Add edges between every pair of nodes that does not yet have one."""
        node_ids = list(self.nodes.keys())
        for i in range(len(node_ids)):
            for j in range(i + 1, len(node_ids)):
                u_id, v_id = node_ids[i], node_ids[j]
                if self.get_edge_by_nodes(u_id, v_id) is None:
                    self.add_edge(u_id, v_id, compute_euclidean_tau(
                        self.nodes[u_id], self.nodes[v_id]
                    ))
