from quantum_solvers.qubo import Qubo
from itertools import product

class VRPProblem:
    def __init__(self, nodes_dict, depot_id, costs_matrix, time_costs_matrix, capacities, customer_ids, customer_demands):
        """
        Initializes the VRPProblem with graph data, including time window information.

        Args:
            nodes_dict (dict): A dictionary mapping node IDs (strings) to Node objects.
                               This provides access to x, y, s, e, l for all nodes.
            depot_id (str): The ID of the depot node.
            costs_matrix (dict): A dictionary of dictionaries representing travel costs
                                 between node IDs: {from_node_id: {to_node_id: cost}}.
            time_costs_matrix (dict): Similar to costs_matrix, but for travel times.
                                      (Often identical to costs_matrix for Euclidean distance).
            capacities (list): A list of vehicle capacities, one entry per vehicle.
            customer_ids (list): A list of customer node IDs (strings), excluding the depot.
            customer_demands (dict): A dictionary mapping customer IDs (strings) to their demands.
        """
        self.nodes_dict = nodes_dict
        self.depot_id = depot_id
        self.costs_matrix = costs_matrix
        self.time_costs_matrix = time_costs_matrix
        self.capacities = capacities
        self.customer_ids = customer_ids
        self.customer_demands = customer_demands

    def get_qubo(self, vehicle_k_limits, only_one_const, order_const, tw_penalty_const):
        """
        Generates the QUBO for the CVRPTW based on a modified formulation to include
        time window penalties.
        Variables are tuples (vehicle_idx, node_id, step_k).
        """
        num_vehicles = len(self.capacities)
        customer_nodes = self.customer_ids # These are the customer IDs (strings)

        qubo = Qubo()

        # ======================================================================
        # CONSTRAINT 1: Each customer is visited exactly once.
        # For each customer j, exactly one (i, j, k) variable must be 1.
        # ======================================================================
        for j_id in customer_nodes:
            variables_for_dest_j = []
            for i in range(num_vehicles):
                k_max = vehicle_k_limits[i]
                for k in range(1, k_max + 1): # Steps k=1 to k_max (customer visits)
                    variables_for_dest_j.append((i, j_id, k))
            qubo.add_only_one_constraint(variables_for_dest_j, only_one_const)

        # ======================================================================
        # CONSTRAINT 2: Each vehicle is in exactly one location at each step.
        # For each vehicle i and step k, exactly one (i, j, k) variable must be 1.
        # This forces vehicles to be "busy" at every step where they are active.
        # ======================================================================
        for i in range(num_vehicles):
            k_max = vehicle_k_limits[i]
            for k in range(1, k_max + 1): # Steps k=1 to k_max
                variables_for_vehicle_step = [(i, j_id, k) for j_id in customer_nodes]
                qubo.add_only_one_constraint(variables_for_vehicle_step, only_one_const)

        # ======================================================================
        # OBJECTIVE FUNCTION C: Minimize travel distance BETWEEN CUSTOMERS.
        # This part of the QUBO minimizes the sum of costs for transitions
        # from one customer to another.
        # Costs to/from the depot are handled classically in VRPSolution.
        # ======================================================================
        for i in range(num_vehicles):
            k_max = vehicle_k_limits[i]
            # Cost between intermediate stops (step k to k+1)
            for k in range(1, k_max): # From step 1 up to k_max-1 (for transitions)
                for j1_id in customer_nodes:
                    for j2_id in customer_nodes:
                        if j1_id == j2_id: continue # Cannot travel from a node to itself in a step
                        var1 = (i, j1_id, k)   # Vehicle i visits j1 at step k
                        var2 = (i, j2_id, k+1) # Vehicle i visits j2 at step k+1
                        
                        cost = self.costs_matrix[j1_id][j2_id]
                        qubo.add((var1, var2), cost * order_const)
        
        # ======================================================================
        # CONSTRAINT 3: Time Window Penalties for Customer-to-Customer Transitions
        # Penalize if a direct transition from j1 to j2 by vehicle i at steps k to k+1
        # would violate j2's latest time window, assuming j1 was visited at its earliest.
        # This is a simplified, but critical, hard-constraint penalty for the QUBO.
        # ======================================================================
        for i in range(num_vehicles):
            k_max = vehicle_k_limits[i]
            for k in range(1, k_max): # From step 1 to k_max-1 (customer to customer transitions)
                for j1_id in customer_nodes:
                    node_j1 = self.nodes_dict[j1_id]
                    for j2_id in customer_nodes:
                        node_j2 = self.nodes_dict[j2_id]
                        if j1_id == j2_id: continue

                        tau_j1_j2 = self.time_costs_matrix[j1_id][j2_id]

                        # Check if earliest possible arrival at j2 (leaving j1 at its earliest)
                        # is already too late for j2's latest time window.
                        # This indicates a hard temporal infeasibility for this sequence.
                        earliest_arrival_at_j2_from_j1 = node_j1.e + node_j1.s + tau_j1_j2
                        if earliest_arrival_at_j2_from_j1 > node_j2.l:
                            var1 = (i, j1_id, k)
                            var2 = (i, j2_id, k+1)
                            qubo.add((var1, var2), tw_penalty_const)

        # ======================================================================
        # CONSTRAINT 4: Time Window Penalties for Depot-to-Customer Transitions (Step 0 to 1)
        # Penalize if a vehicle leaves the depot and arrives at customer j at step 1
        # outside j's time window. Specifically, if it arrives too late.
        # ======================================================================
        depot_node = self.nodes_dict[self.depot_id]
        for i in range(num_vehicles):
            k = 1 # This applies to the first customer visit
            for j_id in customer_nodes:
                node_j = self.nodes_dict[j_id]
                tau_depot_j = self.time_costs_matrix[self.depot_id][j_id]

                # Arrival time at j from depot (assuming depot service time is 0)
                arrival_at_j_from_depot = depot_node.e + tau_depot_j

                # Penalty for late arrival at customer j from depot
                if arrival_at_j_from_depot > node_j.l:
                    # This is a linear penalty on the variable x_{i,j,1}
                    qubo.add(((i, j_id, k), (i, j_id, k)), tw_penalty_const)

                # Optional: Penalty for early arrival (if waiting is costly or strictly forbidden)
                # If arrival_at_j_from_depot < node_j.e:
                #    qubo.add(((i, j_id, k), (i, j_id, k)), tw_penalty_const * (node_j.e - arrival_at_j_from_depot))
                # This is typically not a hard violation, as vehicles can wait.
                # For now, we focus on hard latest time window violations.

        return qubo

