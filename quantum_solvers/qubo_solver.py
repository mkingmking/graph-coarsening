class Qubo:
    def __init__(self):
        self.dict = {}

    def add(self, key, value):
        if isinstance(key, tuple) and len(key) == 2:
            # FIX: The original code failed when 'key' contained mixed-type
            # elements, such as a routing variable (tuple of ints) and a
            # slack variable (tuple starting with a string 's').
            # This try-except block now handles that by sorting based on the
            # string representation of the elements if a TypeError occurs.
            try:
                # Attempt to sort normally first
                sorted_key = tuple(sorted(key))
            except TypeError:
                # If types are mixed, sort based on their string representation
                sorted_key = tuple(sorted(key, key=str))

            self.dict.setdefault(sorted_key, 0)
            self.dict[sorted_key] += value
        else:
            self.dict.setdefault(key, 0)
            self.dict[key] += value

    def add_only_one_constraint(self, variables, penalty):
        """Adds a penalty for sum(vars) != 1."""
        for var in variables:
            self.add((var, var), -penalty)
        for i in range(len(variables)):
            for j in range(i + 1, len(variables)):
                self.add((variables[i], variables[j]), 2 * penalty)

    def add_at_most_one_constraint(self, variables, penalty):
        """Adds a penalty for sum(vars) > 1."""
        for i in range(len(variables)):
            for j in range(i + 1, len(variables)):
                self.add((variables[i], variables[j]), penalty)

    def add_quadratic_equality_constraint(self, linear_expression, constant, penalty):
        """
        Adds a penalty for (linear_expression + constant)^2 to the QUBO.
        'linear_expression' is a list of (coefficient, variable) tuples.
        """
        # Add linear terms: P * (2*c*coeff*var + coeff^2*var)
        for coeff, var in linear_expression:
            self.add((var, var), penalty * (2 * constant * coeff + coeff * coeff))
        
        # Add quadratic terms: P * (2*coeff1*coeff2*var1*var2)
        for i in range(len(linear_expression)):
            for j in range(i + 1, len(linear_expression)):
                coeff1, var1 = linear_expression[i]
                coeff2, var2 = linear_expression[j]
                self.add((var1, var2), penalty * (2 * coeff1 * coeff2))
