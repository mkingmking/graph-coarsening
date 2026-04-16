class VRPSolution:
    def __init__(self, problem, sample, vehicle_k_limits, solution=None):
        self.problem = problem
        self.depot = self.problem.source_depot
        
        if solution is not None:
            self.solution = solution
        else:
            num_vehicles = len(self.problem.capacities)
            temp_routes = {i: [] for i in range(num_vehicles)}

            for var, val in sample.items():
                if val == 1 and isinstance(var, tuple) and len(var) == 3 and isinstance(var[0], int):
                    i, j, k = var
                    if i < num_vehicles:
                        temp_routes[i].append((k, j))

            final_routes = []
            for i in range(num_vehicles):
                sorted_visits = sorted(temp_routes[i], key=lambda x: x[0])
                route = [j for k, j in sorted_visits]
                if route:
                    final_routes.append(route)
            
            repaired = self._repair_solution(final_routes)
            self.solution = self._repair_time_windows(repaired)
    
    def _calculate_arrival_time(self, route, candidate_node=None):
        """
        Helper to calculate the arrival time at the end of a route, 
        optionally including a candidate node appended to the end.
        Returns float('inf') if any node in the chain is late.
        """
        current_time = 0.0
        last_node = self.depot
        
        depot_ready = self.problem.time_windows[self.depot][0]
        current_time = max(current_time, depot_ready)
        
        full_route = route + ([candidate_node] if candidate_node is not None else [])
        
        for node in full_route:
            travel_time = self.problem.time_costs[last_node][node]  # use time_costs
            current_time += travel_time
            
            ready_time, due_date = self.problem.time_windows[node]
            
            if current_time > due_date:
                return float('inf')
            
            current_time = max(current_time, ready_time)
            current_time += self.problem.service_times[node]
            last_node = node
            
        return current_time

    def _repair_solution(self, routes):
        """
        Repairs a solution that may have constraint violations:
        1. Remove duplicate customer visits
        2. Add missing customers
        3. CHECKS TIME WINDOWS before inserting missing customers
        """
        all_customers = set(self.problem.dests)
        visited = set()
        
        # Remove duplicates while preserving first occurrence
        repaired_routes = []
        for route in routes:
            clean_route = []
            for customer in route:
                if customer not in visited:
                    clean_route.append(customer)
                    visited.add(customer)
            if clean_route:
                repaired_routes.append(clean_route)
        
        # Find missing customers
        missing = all_customers - visited
        
        if missing:
            for customer in missing:
                best_route_idx = -1
                best_pos = -1
                best_cost = float('inf')

                if not repaired_routes:
                    repaired_routes.append([customer])
                    continue

                customer_demand = self.problem.weights.get(customer, 0)

                for idx, route in enumerate(repaired_routes):
                    route_demand = sum(self.problem.weights.get(c, 0) for c in route)
                    vehicle_capacity = (self.problem.capacities[idx]
                                        if idx < len(self.problem.capacities)
                                        else self.problem.capacities[0])
                    if route_demand + customer_demand > vehicle_capacity:
                        continue

                    # Try every insertion position: before index 0, between each pair, after last
                    for pos in range(len(route) + 1):
                        candidate_route = route[:pos] + [customer] + route[pos:]

                        # Check TW feasibility of the full candidate route
                        if self._count_route_tw_violations(candidate_route) > 0:
                            continue

                        # Compute insertion cost delta: remove direct edge, add two edges via customer
                        prev_node = self.depot if pos == 0 else route[pos - 1]
                        next_node = self.depot if pos == len(route) else route[pos]
                        delta = (self.problem.costs[prev_node][customer]
                                 + self.problem.costs[customer][next_node]
                                 - self.problem.costs[prev_node][next_node])

                        if delta < best_cost:
                            best_cost = delta
                            best_route_idx = idx
                            best_pos = pos

                if best_route_idx != -1:
                    route = repaired_routes[best_route_idx]
                    repaired_routes[best_route_idx] = route[:best_pos] + [customer] + route[best_pos:]
                else:
                    repaired_routes.append([customer])
        
        return repaired_routes
    
    def _count_route_tw_violations(self, route):
        """Count TW violations along a single route (depot → route → depot)."""
        if not route:
            return 0
        violations = 0
        current_time = max(0.0, self.problem.time_windows[self.depot][0])
        current_time += self.problem.time_costs[self.depot][route[0]]
        ready, due = self.problem.time_windows[route[0]]
        if current_time > due:
            violations += 1
        current_time = max(current_time, ready) + self.problem.service_times[route[0]]
        for idx in range(len(route) - 1):
            current_time += self.problem.time_costs[route[idx]][route[idx + 1]]
            ready, due = self.problem.time_windows[route[idx + 1]]
            if current_time > due:
                violations += 1
            current_time = max(current_time, ready) + self.problem.service_times[route[idx + 1]]
        return violations

    def _repair_time_windows(self, routes):
        """
        Post-processing pass: for each route with TW violations, try all
        permutations of customers (tractable for ≤8 stops) or pairwise swaps
        (for larger routes) to find an ordering with fewer violations.
        Capacity feasibility is preserved — only the visit order changes.
        """
        from itertools import permutations as _perms

        repaired = []
        for route in routes:
            if len(route) <= 1:
                repaired.append(route)
                continue

            best_route = list(route)
            best_viols = self._count_route_tw_violations(best_route)

            if best_viols == 0:
                repaired.append(best_route)
                continue

            if len(route) <= 8:
                # Exhaustive: try every permutation
                for perm in _perms(route):
                    v = self._count_route_tw_violations(list(perm))
                    if v < best_viols:
                        best_viols = v
                        best_route = list(perm)
                        if best_viols == 0:
                            break
            else:
                # Heuristic: repeated pairwise-swap hill-climb
                current = list(route)
                improved = True
                while improved:
                    improved = False
                    for a in range(len(current)):
                        for b in range(a + 1, len(current)):
                            candidate = current[:]
                            candidate[a], candidate[b] = candidate[b], candidate[a]
                            if self._count_route_tw_violations(candidate) < self._count_route_tw_violations(current):
                                current = candidate
                                improved = True
                best_route = current

            repaired.append(best_route)

        return self._repair_inter_route(repaired)

    def _repair_inter_route(self, routes):
        """
        Inter-route repair: repeatedly try moving a customer from a violated
        route to any position in another route, accepting moves that reduce
        total TW violations without breaking capacity. Runs until no improvement.
        """
        routes = [list(r) for r in routes]

        def total_violations(rts):
            return sum(self._count_route_tw_violations(r) for r in rts)

        improved = True
        while improved:
            improved = False
            best_delta = 0          # must strictly improve
            best_move = None        # (src_idx, src_pos, dst_idx, dst_pos)

            viol_total = total_violations(routes)
            if viol_total == 0:
                break

            for src_idx, src_route in enumerate(routes):
                if not src_route:
                    continue
                src_viols = self._count_route_tw_violations(src_route)
                if src_viols == 0:
                    continue  # this route is already fine

                for src_pos, customer in enumerate(src_route):
                    # Route after removing this customer
                    src_without = src_route[:src_pos] + src_route[src_pos + 1:]
                    src_viols_after = self._count_route_tw_violations(src_without)

                    customer_demand = self.problem.weights.get(customer, 0)

                    for dst_idx, dst_route in enumerate(routes):
                        if dst_idx == src_idx:
                            continue

                        # Capacity check
                        cap = (self.problem.capacities[dst_idx]
                               if dst_idx < len(self.problem.capacities)
                               else self.problem.capacities[0])
                        dst_load = sum(self.problem.weights.get(c, 0) for c in dst_route)
                        if dst_load + customer_demand > cap:
                            continue

                        for dst_pos in range(len(dst_route) + 1):
                            dst_with = dst_route[:dst_pos] + [customer] + dst_route[dst_pos:]
                            dst_viols_after = self._count_route_tw_violations(dst_with)

                            # Violation delta: negative means improvement
                            delta = (src_viols_after - src_viols) + dst_viols_after
                            if delta < best_delta:
                                best_delta = delta
                                best_move = (src_idx, src_pos, dst_idx, dst_pos)

            if best_move is not None:
                src_idx, src_pos, dst_idx, dst_pos = best_move
                customer = routes[src_idx][src_pos]
                routes[src_idx] = routes[src_idx][:src_pos] + routes[src_idx][src_pos + 1:]
                routes[dst_idx] = routes[dst_idx][:dst_pos] + [customer] + routes[dst_idx][dst_pos:]
                improved = True

        # Remove empty routes
        return [r for r in routes if r]

    def check(self):
        """
        Validates the solution for:
          - No duplicate customer visits
          - Capacity constraints per vehicle
          - Time window constraints per route
          - All customers are visited
        """
        capacities  = self.problem.capacities
        weights     = self.problem.weights
        time_windows  = self.problem.time_windows
        service_times = self.problem.service_times
        time_costs    = self.problem.time_costs

        visited_customers = set()
        for i, route in enumerate(self.solution):

            # ── Duplicate check ───────────────────────────────────────────
            for customer in route:
                if customer in visited_customers:
                    return False
                visited_customers.add(customer)
            
            # ── Capacity check ────────────────────────────────────────────
            current_load = sum(weights.get(dest, 0) for dest in route)
            if i < len(capacities) and current_load > capacities[i]:
                return False

            if not route:
                continue
            
            # ── Time window check ─────────────────────────────────────────
            current_time = 0.0
            
            depot_ready = time_windows[self.depot][0]
            current_time = max(current_time, depot_ready)
            
            # Depot → first stop
            current_time += time_costs[self.depot][route[0]]
            
            ready_time, due_date = time_windows[route[0]]
            if current_time > due_date:
                return False
            current_time = max(current_time, ready_time)
            current_time += service_times[route[0]]

            # Each subsequent stop
            for stop_idx in range(len(route) - 1):
                from_node = route[stop_idx]
                to_node   = route[stop_idx + 1]

                current_time += time_costs[from_node][to_node]
                
                ready_time, due_date = time_windows[to_node]
                if current_time > due_date:
                    return False
                current_time = max(current_time, ready_time)
                current_time += service_times[to_node]

        # ── Completeness check ────────────────────────────────────────────
        required_customers = set(self.problem.dests)
        if visited_customers != required_customers:
            missing = required_customers - visited_customers
            print(f"Error: Solution is incomplete. Missing customers: {missing}")
            return False

        return True

    def total_cost(self):
        total_cost = 0
        for route in self.solution:
            if not route: continue
            
            route_cost = self.problem.costs[self.depot][route[0]]
            
            for i in range(len(route) - 1):
                route_cost += self.problem.costs[route[i]][route[i+1]]
            
            route_cost += self.problem.costs[route[-1]][self.depot]
            
            total_cost += route_cost
        return total_cost

    def description(self):
        print("Solution Routes:")
        for i, route in enumerate(self.solution):
            path_str = " -> ".join(map(str, [self.depot] + route + [self.depot]))
            print(f"  Vehicle {i}: {path_str}")
        print(f"\nTotal Cost: {self.total_cost():.2f}")
        print(f"Is Solution Valid (Capacity/TW): {self.check()}")
