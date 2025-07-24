# node.py

import math

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
        t (float): Central time, calculated as (e + l) / 2.
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
        # Calculate central time (t_i) considering the effective window for service START
        # (earliest start + (latest start - service time)) / 2
        self.t = (e + (l - s)) / 2 if (l - s) >= 0 else e
        self.is_super_node = is_super_node
        self.original_nodes = original_nodes if original_nodes is not None else [id]

    def __repr__(self):
        return (f"Node(ID={self.id}, Coords=({self.x:.2f},{self.y:.2f}), S={self.s:.2f}, "
                f"TW=[{self.e:.2f},{self.l:.2f}], T={self.t:.2f}, Demand={self.demand:.2f}, Super={self.is_super_node})")

def compute_euclidean_tau(node1: Node, node2: Node) -> float:
    """
    Computes the Euclidean travel time (distance) between two nodes.
    This helper function is placed here as it's fundamental to node interactions.
    """
    return math.sqrt((node1.x - node2.x)**2 + (node1.y - node2.y)**2)
