import random

class MockDWaveSolvers:
    """
    A mock class for DWaveSolvers_modified to allow local testing
    without actual D-Wave access.
    This mock will return a dummy solution for a given QUBO.
    It does NOT actually solve the QUBO for minimum energy.
    """
    @staticmethod
    def solve_qubo(qubo_instance, solver_type='simulated', limit=1, num_reads=50):
        """
        Simulates solving a QUBO. For demonstration, it returns a simple dummy solution.
        In a real scenario, this would call D-Wave or a local sampler.
        
        Args:
            qubo_instance: An instance of the Qubo class.
            solver_type (str): Ignored in mock.
            limit (int): Ignored in mock, always returns 1 sample if any variables.
            num_reads (int): Ignored in mock.

        Returns:
            list: A list containing one dictionary representing a sample solution,
                  or an empty list if no variables are in the QUBO.
        """
        # Get all unique variables from the QUBO dictionary keys
        all_vars = set()
        for (u, v) in qubo_instance.get_dict().keys():
            all_vars.add(u)
            all_vars.add(v)
        
        # Filter for actual VRP variables (vehicle_idx, customer_id, step_k)
        # Now, customer_id is a string, not an integer index.
        qubo_variables = [v for v in all_vars if isinstance(v, tuple) and len(v) == 3]

        if not qubo_variables:
            return [] # No variables to set, return empty

        # Create a dummy sample: initialize all variables to 0
        sample = {var: 0 for var in qubo_variables}
        
        # Attempt to create a "sensible" dummy solution for VRP:
        # Try to assign each customer to a vehicle at step 1, if possible.
        # This is very basic and won't be optimal or necessarily feasible,
        # but it creates a non-empty route for testing the flow.

        # Group variables by customer to ensure each customer is assigned once
        customers_to_assign = {v[1] for v in qubo_variables} # Unique customer IDs (strings)
        assigned_customers = set()

        for customer_id in customers_to_assign:
            # Find all variables for this customer
            customer_vars = [v for v in qubo_variables if v[1] == customer_id]
            
            # Try to assign it to a vehicle at step 1 if available, otherwise any step
            step_1_vars_for_customer = [v for v in customer_vars if v[2] == 1]

            if step_1_vars_for_customer:
                # Pick a random vehicle for this customer at step 1
                chosen_var = random.choice(step_1_vars_for_customer)
                if chosen_var not in sample or sample[chosen_var] == 0: # Avoid overwriting if already picked
                    sample[chosen_var] = 1
                    assigned_customers.add(customer_id)
            elif customer_vars:
                # If no step 1, pick any available step for this customer
                chosen_var = random.choice(customer_vars)
                if chosen_var not in sample or sample[chosen_var] == 0:
                    sample[chosen_var] = 1
                    assigned_customers.add(customer_id)
        
        # Ensure that if a customer couldn't be assigned by the above logic,
        # we still try to assign it to *some* vehicle and step if possible,
        # to make the solution less empty for testing.
        for customer_id in customers_to_assign:
            if customer_id not in assigned_customers:
                customer_vars = [v for v in qubo_variables if v[1] == customer_id]
                if customer_vars:
                    chosen_var = random.choice(customer_vars)
                    sample[chosen_var] = 1
                    assigned_customers.add(customer_id)


        return [sample]

