# savings_solver.py
import math
from .graph import Graph
from .node import Node
from .utils import compute_euclidean_tau, calculate_route_metrics

class SavingsSolver:
    """
    Implements the Clarke and Wright Savings Algorithm for VRPTW.
    Generates multiple routes.
    """
    def __init__(self, graph: Graph, depot_id: str, vehicle_capacity: float):
        self.graph = graph
        self.depot_id = depot_id
        self.vehicle_capacity = vehicle_capacity

    def _calculate_savings(self) -> list:
        savings = []
        customer_ids = [node_id for node_id in self.graph.nodes if node_id != self.depot_id]
        depot_node = self.graph.nodes[self.depot_id]

        for i in range(len(customer_ids)):
            for j in range(i + 1, len(customer_ids)):
                id_i = customer_ids[i]
                id_j = customer_ids[j]
                node_i = self.graph.nodes[id_i]
                node_j = self.graph.nodes[id_j]

                tau_di = compute_euclidean_tau(depot_node, node_i)
                tau_jd = compute_euclidean_tau(node_j, depot_node)
                tau_ij = compute_euclidean_tau(node_i, node_j)
                
                saving = tau_di + tau_jd - tau_ij
                savings.append((saving, id_i, id_j))
        
        savings.sort(key=lambda x: x[0], reverse=True)
        return savings

    def _check_merge_feasibility(self, route1: list, route2: list, merge_point_i: str, merge_point_j: str) -> bool:
        idx_i = -1
        for k in range(len(route1) - 1):
            if route1[k] == merge_point_i:
                idx_i = k
                break
        
        idx_j = -1
        for k in range(1, len(route2)):
            if route2[k] == merge_point_j:
                idx_j = k
                break

        if idx_i == -1 or idx_j == -1:
            return False

        candidate_route = route1[:idx_i+1] + route2[idx_j:]
        
        if candidate_route[0] != self.depot_id:
            candidate_route.insert(0, self.depot_id)
        if candidate_route[-1] != self.depot_id:
            candidate_route.append(self.depot_id)

        current_time = self.graph.nodes[self.depot_id].e
        current_load = 0.0

        for k in range(len(candidate_route) - 1):
            from_node_id = candidate_route[k]
            to_node_id = candidate_route[k+1]

            from_node = self.graph.nodes[from_node_id]
            to_node = self.graph.nodes[to_node_id]

            if to_node_id != self.depot_id:
                current_load += to_node.demand
                if current_load > self.vehicle_capacity:
                    return False

            travel_time = compute_euclidean_tau(from_node, to_node)
            arrival_time_at_to_node = current_time + travel_time
            
            service_start_time_at_to_node = max(arrival_time_at_to_node, to_node.e)

            if service_start_time_at_to_node > to_node.l:
                return False

            current_time = service_start_time_at_to_node + to_node.s
        
        last_customer_node = self.graph.nodes[candidate_route[-2]]
        depot_node = self.graph.nodes[self.depot_id]
        travel_time_to_depot = compute_euclidean_tau(last_customer_node, depot_node)
        final_arrival_at_depot = current_time + travel_time_to_depot
        
        if final_arrival_at_depot > depot_node.l:
            return False

        return True

    def solve(self) -> tuple[list, dict]:
        print(f"\n--- Starting Savings Solver on graph with depot {self.depot_id} ---")

        routes = {}
        customer_ids = [node_id for node_id in self.graph.nodes if node_id != self.depot_id]
        for cust_id in customer_ids:
            routes[cust_id] = [self.depot_id, cust_id, self.depot_id]
        
        customer_to_route_map = {cust_id: cust_id for cust_id in customer_ids}

        print(f"  Initial routes: {len(customer_ids)} individual routes.")

        savings = self._calculate_savings()
        print(f"  Calculated {len(savings)} potential savings.")

        merged_any_this_iteration = True
        while merged_any_this_iteration:
            merged_any_this_iteration = False
            for saving_value, id_i, id_j in savings:
                if id_i not in customer_to_route_map or id_j not in customer_to_route_map:
                    continue

                route_id_i = customer_to_route_map[id_i]
                route_id_j = customer_to_route_map[id_j]

                if route_id_i == route_id_j:
                    continue

                route_i = routes[route_id_i]
                route_j = routes[route_id_j]
                
                can_merge_i_j = (route_i[-2] == id_i and route_j[1] == id_j)
                can_merge_j_i = (route_j[-2] == id_j and route_i[1] == id_i)

                if not (can_merge_i_j or can_merge_j_i):
                    continue

                proposed_merged_route = None
                if can_merge_i_j:
                    temp_route = route_i[:-1] + route_j[1:]
                    if self._check_merge_feasibility(route_i, route_j, id_i, id_j):
                        proposed_merged_route = temp_route
                
                if proposed_merged_route is None and can_merge_j_i:
                    temp_route = route_j[:-1] + route_i[1:]
                    if self._check_merge_feasibility(route_j, route_i, id_j, id_i):
                        proposed_merged_route = temp_route

                if proposed_merged_route:
                    print(f"  Merging routes for {id_i} and {id_j} with saving {saving_value:.2f}")
                    
                    new_route_id = f"R_{id_i}_{id_j}"
                    routes[new_route_id] = proposed_merged_route
                    
                    for customer_in_old_route in route_i[1:-1]:
                        customer_to_route_map[customer_in_old_route] = new_route_id
                    for customer_in_old_route in route_j[1:-1]:
                        customer_to_route_map[customer_in_old_route] = new_route_id
                    
                    del routes[route_id_i]
                    del routes[route_id_j]
                    
                    merged_any_this_iteration = True
                    break
            
            if not merged_any_this_iteration:
                break

        final_routes_list = list(routes.values())
        print(f"--- Savings Solver Finished. Found {len(final_routes_list)} routes. ---")
        
        metrics = calculate_route_metrics(self.graph, final_routes_list, self.depot_id, self.vehicle_capacity)
        return final_routes_list, metrics

