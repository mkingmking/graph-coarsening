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

