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
        # Central time calculation: (earliest start + (latest start - service time)) / 2
        # This considers the effective time window for the *start* of service.
        self.t = (e + (l - s)) / 2 if (l - s) >= 0 else e
        self.is_super_node = is_super_node
        self.original_nodes = original_nodes if original_nodes is not None else [id]

    def __repr__(self):
        return (f"Node(ID={self.id}, Coords=({self.x:.2f},{self.y:.2f}), S={self.s:.2f}, "
                f"TW=[{self.e:.2f},{self.l:.2f}], T={self.t:.2f}, Demand={self.demand:.2f}, Super={self.is_super_node})")

