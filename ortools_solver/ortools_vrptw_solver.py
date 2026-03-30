"""
OR-Tools CVRPTW Solver
----------------------
Wraps Google OR-Tools' vehicle routing library to solve CVRPTW instances
using the same Graph-based interface as GreedySolver and SavingsSolver.

Returns routes in the format [[depot, c1, c2, ..., depot], ...] and a
metrics dict produced by calculate_route_metrics, so it plugs directly
into the existing pipeline (coarsening, inflation, comparisons).
"""

from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

from ..graph import Graph, compute_euclidean_tau
from ..utils import calculate_route_metrics


class ORToolsVRPTWSolver:
    """
    Solves CVRPTW using Google OR-Tools' CP-based routing solver.

    Parameters
    ----------
    graph : Graph
        The (possibly coarsened) problem graph.
    depot_id : str
        Node ID of the depot.
    vehicle_capacity : float
        Maximum load per vehicle.
    time_limit_seconds : int
        Wall-clock time limit for the local-search phase (default 30 s).
        Increase for better solutions on large instances.
    """

    def __init__(
        self,
        graph: Graph,
        depot_id: str,
        vehicle_capacity: float,
        time_limit_seconds: int = None,
        solution_limit: int = None,
        vehicle_fixed_cost: int = 100_000,
    ):
        """
        vehicle_fixed_cost : unscaled fixed cost added per active vehicle.
            The standard VRPTW objective is lexicographic: minimise vehicles
            first, then distance. Setting this larger than the maximum possible
            single-route distance achieves that. For Solomon (coords in [0,100],
            max route ~1400 units), 100_000 dominates any distance difference.
            Set to 0 for distance-only objective.

        Stopping criteria (pick one or both; first reached wins):
        - time_limit_seconds : hard wall-clock cap.
        - solution_limit     : stop after N improving solutions (OR-Tools exits
                               immediately so perf_counter gives true time).
        If neither is set, a 300 s safety cap is applied automatically.
        """
        self.graph = graph
        self.depot_id = depot_id
        self.vehicle_capacity = vehicle_capacity
        self.time_limit_seconds = time_limit_seconds
        self.solution_limit = solution_limit
        self.vehicle_fixed_cost = vehicle_fixed_cost

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def solve(
        self,
        initial_routes: list = None,
    ) -> tuple[list, dict]:
        """
        Solves the CVRPTW and returns routes + metrics.

        Parameters
        ----------
        initial_routes : list[list[str]], optional
            Warm-start routes on this graph's node IDs (depot may be
            included or omitted — it is stripped before passing to OR-Tools).
            When provided, OR-Tools starts local search from this solution
            instead of building one from scratch.

        Returns
        -------
        routes : list[list[str]]
            Each inner list is a full route starting and ending at the depot:
            [depot_id, cust1, cust2, ..., depot_id]
        metrics : dict
            Same keys as calculate_route_metrics output.
        """
        # ---- 1. Build ordered node list (depot at index 0) ----
        node_ids = [self.depot_id] + [
            nid for nid in self.graph.nodes if nid != self.depot_id
        ]
        id_to_idx = {nid: i for i, nid in enumerate(node_ids)}
        n = len(node_ids)

        # OR-Tools requires integer costs. Solomon coordinates fit in
        # [0, 100] so Euclidean distances are at most ~141; scale ×100
        # keeps precision while staying well within 32-bit int range.
        SCALE = 100

        # ---- 2. Pre-compute integer distance matrix ----
        dist_matrix = [
            [
                round(
                    compute_euclidean_tau(
                        self.graph.nodes[node_ids[i]],
                        self.graph.nodes[node_ids[j]],
                    )
                    * SCALE
                )
                if i != j
                else 0
                for j in range(n)
            ]
            for i in range(n)
        ]

        # ---- 3. Time windows and service times (scaled) ----
        time_windows = [
            (
                round(self.graph.nodes[nid].e * SCALE),
                round(self.graph.nodes[nid].l * SCALE),
            )
            for nid in node_ids
        ]
        service_times = [round(self.graph.nodes[nid].s * SCALE) for nid in node_ids]

        # ---- 4. Demands and capacity (Solomon uses integers) ----
        demands = [int(self.graph.nodes[nid].demand) for nid in node_ids]
        capacity = int(self.vehicle_capacity)

        # Upper bound on vehicles: one per customer
        num_vehicles = n - 1
        depot_idx = 0

        # ---- 5. Build OR-Tools routing model ----
        manager = pywrapcp.RoutingIndexManager(n, num_vehicles, depot_idx)
        routing = pywrapcp.RoutingModel(manager)

        # Arc cost = travel distance
        def _distance_cb(from_idx, to_idx):
            return dist_matrix[manager.IndexToNode(from_idx)][
                manager.IndexToNode(to_idx)
            ]

        transit_cb_idx = routing.RegisterTransitCallback(_distance_cb)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_cb_idx)

        # Vehicle fixed cost — makes OR-Tools minimise vehicles first, then
        # distance (the standard VRPTW lexicographic objective).
        # Without this, OR-Tools minimises arc distance only and happily
        # assigns every customer to its own vehicle (which minimises empty
        # travel per customer but completely ignores vehicle count).
        # Value is in scaled units (×SCALE); 100_000 * SCALE safely dominates
        # any realistic route distance for Solomon instances.
        if self.vehicle_fixed_cost > 0:
            routing.SetFixedCostOfAllVehicles(self.vehicle_fixed_cost * SCALE)

        # ---- 6. Time dimension (travel + service at origin node) ----
        def _time_cb(from_idx, to_idx):
            fn = manager.IndexToNode(from_idx)
            tn = manager.IndexToNode(to_idx)
            return dist_matrix[fn][tn] + service_times[fn]

        time_cb_idx = routing.RegisterTransitCallback(_time_cb)

        # Horizon large enough for any feasible schedule.
        # Use max of both window bounds in case coarsening inverted some windows.
        horizon = int(
            max(max(tw) for tw in time_windows)
            + max(service_times)
            + max(dist_matrix[0])
        )

        routing.AddDimension(
            time_cb_idx,
            horizon,   # max waiting / slack per stop
            horizon,   # max cumulative time
            False,     # do not force vehicles to start at time 0
            "Time",
        )
        time_dim = routing.GetDimensionOrDie("Time")

        # Apply time-window bounds to every node.
        # After coarsening, super-node windows can be very tight or
        # even inverted (e > l) due to the merging arithmetic.
        # Clamp e to [0, l] so OR-Tools never receives an invalid range.
        for node_idx in range(n):
            idx = manager.NodeToIndex(node_idx)
            tw_e, tw_l = time_windows[node_idx]
            tw_e = max(0, min(tw_e, tw_l))
            time_dim.CumulVar(idx).SetRange(tw_e, tw_l)

        # ---- 7. Capacity dimension ----
        def _demand_cb(from_idx):
            return demands[manager.IndexToNode(from_idx)]

        demand_cb_idx = routing.RegisterUnaryTransitCallback(_demand_cb)
        routing.AddDimensionWithVehicleCapacity(
            demand_cb_idx,
            0,                        # no slack
            [capacity] * num_vehicles,
            True,                     # start cumul at zero
            "Capacity",
        )

        # ---- 8. Search parameters ----
        params = pywrapcp.DefaultRoutingSearchParameters()
        # SAVINGS (Clarke-Wright) builds initial routes by merging depot
        # round-trips, which naturally minimises vehicle count — critical for
        # matching the lexicographic VRPTW objective (vehicles first, distance
        # second). PATH_CHEAPEST_ARC tends to create many small routes that
        # GLS struggles to merge even with a high vehicle fixed cost.
        params.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.SAVINGS
        )
        params.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        )
        params.log_search = False

        # Apply stopping criteria. solution_limit is preferred for timing
        # experiments because OR-Tools exits immediately when reached, so
        # the wall-clock time accurately reflects actual solver effort.
        if self.solution_limit is not None:
            params.solution_limit = self.solution_limit
        if self.time_limit_seconds is not None:
            params.time_limit.seconds = self.time_limit_seconds
        if self.solution_limit is None and self.time_limit_seconds is None:
            params.time_limit.seconds = 300  # safety cap

        # ---- 9. Optionally inject a warm-start solution ----
        initial_assignment = None
        if initial_routes:
            # Strip depot from each route, map IDs to node indices,
            # and drop any IDs not present in this graph (safety guard).
            known = set(node_ids)
            idx_routes = [
                [
                    id_to_idx[nid]
                    for nid in route
                    if nid != self.depot_id and nid in known
                ]
                for route in initial_routes
                if any(nid != self.depot_id and nid in known for nid in route)
            ]
            if idx_routes:
                initial_assignment = routing.ReadAssignmentFromRoutes(
                    idx_routes, True  # ignore_inactive_nodes=True
                )

        # ---- 10. Solve ----
        if initial_assignment is not None:
            solution = routing.SolveFromAssignmentWithParameters(
                initial_assignment, params
            )
        else:
            solution = routing.SolveWithParameters(params)

        # OR-Tools status codes:
        #   1 = ROUTING_SUCCESS (feasible, may not be optimal)
        #   2 = ROUTING_PARTIAL_SUCCESS_LOCAL_OPTIMUM_NOT_REACHED
        #   3 = ROUTING_FAIL (no solution found)
        #   4 = ROUTING_FAIL_TIMEOUT
        #   6 = ROUTING_OPTIMAL (proven optimal)
        STATUS_LABELS = {1: "feasible", 2: "feasible", 3: "no_solution",
                         4: "timeout", 6: "optimal"}
        status_code = routing.status()
        status_label = STATUS_LABELS.get(status_code, f"status_{status_code}")

        if solution is None or status_code in (3,):
            print(f"[ORToolsVRPTWSolver] No solution found (status={status_label}).")
            metrics = calculate_route_metrics(
                self.graph, [], self.depot_id, self.vehicle_capacity
            )
            metrics["solver_status"] = status_label
            return [], metrics

        # ---- 11. Extract routes ----
        routes = []
        for vehicle_id in range(num_vehicles):
            idx = routing.Start(vehicle_id)
            route = []
            while not routing.IsEnd(idx):
                route.append(node_ids[manager.IndexToNode(idx)])
                idx = solution.Value(routing.NextVar(idx))
            route.append(node_ids[manager.IndexToNode(idx)])  # closing depot

            # Skip empty vehicle legs (depot → depot only)
            if len(route) > 2:
                routes.append(route)

        metrics = calculate_route_metrics(
            self.graph, routes, self.depot_id, self.vehicle_capacity
        )
        metrics["solver_status"] = status_label
        return routes, metrics
