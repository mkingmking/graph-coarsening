class Qubo:
    def __init__(self):
        self.dict = {}

    def add(self, key, value):
        if isinstance(key, tuple) and len(key) == 2:
            try:
                sorted_key = tuple(sorted(key))
            except TypeError:
                sorted_key = tuple(sorted(key, key=str))

            self.dict.setdefault(sorted_key, 0)
            self.dict[sorted_key] += value
        else:
            self.dict.setdefault(key, 0)
            self.dict[key] += value

    def add_only_one_constraint(self, variables, penalty):
        """
        CORRECTED: Adds a penalty for sum(vars) != 1.
        
        The paper's A() function: A(y1,...,yn) = 2*sum(yi*yj) - sum(yi)
        Minimum is -1 when exactly one variable is 1.
        
        To enforce this, we add: penalty * (sum(vars) - 1)^2
        Which expands to: penalty * [sum(vi^2) - 2*sum(vi) + sum(2*vi*vj)]
        
        Since vi ∈ {0,1}, we have vi^2 = vi, so:
        penalty * [sum(vi) - 2*sum(vi) + sum(2*vi*vj)]
        = penalty * [-sum(vi) + sum(2*vi*vj)]
        """
        # Linear terms: -penalty for each variable
        for var in variables:
            self.add((var, var), -penalty)
        
        # Quadratic terms: 2*penalty for each pair
        for i in range(len(variables)):
            for j in range(i + 1, len(variables)):
                self.add((variables[i], variables[j]), 2 * penalty)

    def add_at_most_one_constraint(self, variables, penalty):
        """
        Adds a penalty for sum(vars) > 1.
        This only penalizes having MORE than one variable set to 1.
        
        We want to penalize: sum(vi) * (sum(vi) - 1)
        Which expands to: sum(vi*vj) for i!=j
        """
        for i in range(len(variables)):
            for j in range(i + 1, len(variables)):
                self.add((variables[i], variables[j]), penalty)

    def add_quadratic_equality_constraint(self, linear_expression, constant, penalty):
        """
        Adds a penalty for (linear_expression + constant)^2 to the QUBO.
        'linear_expression' is a list of (coefficient, variable) tuples.
        
        CORRECTED: This now properly expands (sum(coeff_i * var_i) + c)^2
        """
        # Expand (sum(ai*xi) + c)^2 = sum(ai^2*xi^2) + 2c*sum(ai*xi) + c^2 + sum(2*ai*aj*xi*xj)
        
        # Linear terms: ai^2*xi + 2c*ai*xi = (ai^2 + 2c*ai)*xi
        # Since xi^2 = xi for binary variables
        for coeff, var in linear_expression:
            self.add((var, var), penalty * (coeff * coeff + 2 * constant * coeff))
        
        # Quadratic terms: 2*ai*aj*xi*xj
        for i in range(len(linear_expression)):
            for j in range(i + 1, len(linear_expression)):
                coeff1, var1 = linear_expression[i]
                coeff2, var2 = linear_expression[j]
                self.add((var1, var2), penalty * (2 * coeff1 * coeff2))
        
        # Constant term c^2 doesn't affect optimization (no variables), so we can ignore it