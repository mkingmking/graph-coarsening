# or_tools_solver.py
import math
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
from .graph import Graph
from .node import Node
from .utils import compute_euclidean_tau, calculate_route_metrics

class ORToolsSolver:
    """
    Implements a VRPTW solver using Google OR-Tools Routing Library.
    """
    def __init__(self, graph: Graph, depot_id: str, vehicle_capacity: float, num_vehicles: int):
        self.graph = graph
        self.depot_id = depot_id
        self.vehicle_capacity = vehicle_capacity
        self.num_vehicles = num_vehicles

        # Create a mapping from node IDs to OR-Tools internal indices
        # The depot must be the first node in the locations list for OR-Tools' default behavior
        self.node_ids_to_indices = {}
        self.indices_to_node_ids = []

        # Ensure depot is at index 0
        self.node_ids_to_indices[self.depot_id] = 0
        self.indices_to_node_ids.append(self.depot_id)

        # Add other customers
        for node_id in self.graph.nodes:
            if node_id != self.depot_id:
                self.node_ids_to_indices[node_id] = len(self.indices_to_node_ids)
                self.indices_to_node_ids.append(node_id)
        
        self.depot_index = self.node_ids_to_indices[self.depot_id]

        # Precompute distance matrix, demands, and time windows
        self.distance_matrix = self._create_distance_matrix()
        self.demands = self._create_demands_list()
        self.time_windows = self._create_time_windows_list()
        self.service_times = self._create_service_times_list()

    def _create_distance_matrix(self) -> list[list[int]]:
        """Creates a distance matrix (travel times) for OR-Tools."""
        num_locations = len(self.indices_to_node_ids)
        matrix = [[0 for _ in range(num_locations)] for _ in range(num_locations)]
        for i in range(num_locations):
            for j in range(num_locations):
                if i == j:
                    matrix[i][j] = 0
                else:
                    node1_id = self.indices_to_node_ids[i]
                    node2_id = self.indices_to_node_ids[j]
                    node1 = self.graph.nodes[node1_id]
                    node2 = self.graph.nodes[node2_id]
                    matrix[i][j] = int(compute_euclidean_tau(node1, node2)) # OR-Tools prefers integers
        return matrix

    def _create_demands_list(self) -> list[int]:
        """Creates a list of demands for OR-Tools."""
        demands = [0] * len(self.indices_to_node_ids)
        for i, node_id in enumerate(self.indices_to_node_ids):
            demands[i] = int(self.graph.nodes[node_id].demand)
        return demands

    def _create_time_windows_list(self) -> list[tuple[int, int]]:
        """Creates a list of time windows (earliest, latest) for OR-Tools."""
        time_windows = [(0, 0)] * len(self.indices_to_node_ids)
        for i, node_id in enumerate(self.indices_to_node_ids):
            node = self.graph.nodes[node_id]
            time_windows[i] = (int(node.e), int(node.l))
        return time_windows

    def _create_service_times_list(self) -> list[int]:
        """Creates a list of service times for OR-Tools."""
        service_times = [0] * len(self.indices_to_node_ids)
        for i, node_id in enumerate(self.indices_to_node_ids):
            service_times[i] = int(self.graph.nodes[node_id].s)
        return service_times

    def solve(self) -> tuple[list, dict]:
        """
        Solves the VRPTW using Google OR-Tools.
        
        Returns:
            tuple: A tuple containing:
                - list: A list of generated routes (each route is a list of node IDs).
                - dict: A dictionary of aggregated metrics for all routes.
        """
        print(f"\n--- Starting OR-Tools Solver on graph with depot {self.depot_id} ---")

        manager = pywrapcp.RoutingIndexManager(
            len(self.distance_matrix), self.num_vehicles, self.depot_index
        )
        model = pywrapcp.RoutingModel(manager)

        # Create distance callback
        def distance_callback(from_index, to_index):
            from_node_id = manager.IndexToNode(from_index)
            to_node_id = manager.IndexToNode(to_index)
            return self.distance_matrix[from_node_id][to_node_id]

        transit_callback_index = model.RegisterTransitCallback(distance_callback)
        model.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

        # Add Capacity constraint
        def demand_callback(from_index):
            from_node_id = manager.IndexToNode(from_index)
            return self.demands[from_node_id]

        demand_callback_index = model.RegisterUnaryTransitCallback(demand_callback)
        model.AddDimensionWithVehicleCapacity(
            demand_callback_index,
            0,  # null capacity slack
            [int(self.vehicle_capacity)] * self.num_vehicles,  # vehicle capacities (corrected line)
            True,  # start cumul to zero
            'Capacity'
        )

        # Add Time Window constraint
        def time_callback(from_index, to_index):
            from_node_id = manager.IndexToNode(from_index)
            to_node_id = manager.IndexToNode(to_index)
            # Travel time + service time at the 'from' node
            return self.distance_matrix[from_node_id][to_node_id] + self.service_times[from_node_id]

        time_callback_index = model.RegisterTransitCallback(time_callback)
        model.AddDimension(
            time_callback_index,
            30000,  # big slack time (max waiting time at a node)
            30000,  # big max time (max total route duration)
            False,  # Don't start cumul to zero (time starts at depot's earliest)
            'Time'
        )
        time_dimension = model.GetDimensionOrDie('Time')

        for location_idx, time_window in enumerate(self.time_windows):
            if location_idx == self.depot_index: # Depot's time window for start and end
                for vehicle_id in range(self.num_vehicles):
                    time_dimension.SetCumulVarSoftUpperBound(
                        model.Start(vehicle_id), time_window[1], 100000 # High penalty for starting late
                    )
                    time_dimension.SetCumulVarSoftUpperBound(
                        model.End(vehicle_id), time_window[1], 100000 # High penalty for ending late
                    )
            else: # Customer time windows
                time_dimension.SetCumulVarRange( # Corrected line: pass location_idx directly
                    location_idx, time_window[0], time_window[1]
                )

        # Penalize unvisited nodes (makes sure all customers are visited if possible)
        for customer_index in range(len(self.distance_matrix)):
            if customer_index == self.depot_index:
                continue
            model.AddDisjunction([manager.NodeToIndex(customer_index)], 1000000) # High penalty for not visiting

        # Set search parameters
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        )
        search_parameters.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        )
        search_parameters.time_limit.seconds = 30 # Set a time limit for larger problems

        # Solve the problem
        solution = model.SolveWithParameters(search_parameters)

        extracted_routes = []
        if solution:
            print("  OR-Tools Solution Found!")
            for vehicle_id in range(self.num_vehicles):
                route = []
                index = model.Start(vehicle_id)
                if solution.Value(model.NextVar(index)) != manager.NodeToIndex(self.depot_index): # Check if vehicle actually moved
                    while not model.IsEnd(index):
                        node_index = manager.IndexToNode(index)
                        route.append(self.indices_to_node_ids[node_index])
                        index = solution.Value(model.NextVar(index))
                    node_index = manager.IndexToNode(index) # Add the last node (depot)
                    route.append(self.indices_to_node_ids[node_index])
                    
                    if len(route) > 2: # Only add routes that actually served customers
                        extracted_routes.append(route)
                else:
                    # Vehicle did not move from depot
                    pass
        else:
            print("  OR-Tools did not find a solution.")

        print(f"--- OR-Tools Solver Finished ---")
        
        metrics = calculate_route_metrics(self.graph, extracted_routes, self.depot_id, self.vehicle_capacity)
        return extracted_routes, metrics
