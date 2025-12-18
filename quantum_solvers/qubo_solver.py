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
        
        # Linear terms: -penalty for each variable
        for var in variables:
            self.add((var, var), -penalty)
        
        # Quadratic terms: 2*penalty for each pair
        for i in range(len(variables)):
            for j in range(i + 1, len(variables)):
                self.add((variables[i], variables[j]), 2 * penalty)

    def add_at_most_one_constraint(self, variables, penalty):
        
        for i in range(len(variables)):
            for j in range(i + 1, len(variables)):
                self.add((variables[i], variables[j]), penalty)

    def add_quadratic_equality_constraint(self, linear_expression, constant, penalty):
        
        for coeff, var in linear_expression:
            self.add((var, var), penalty * (coeff * coeff + 2 * constant * coeff))
        
        # Quadratic terms: 2*ai*aj*xi*xj
        for i in range(len(linear_expression)):
            for j in range(i + 1, len(linear_expression)):
                coeff1, var1 = linear_expression[i]
                coeff2, var2 = linear_expression[j]
                self.add((var1, var2), penalty * (2 * coeff1 * coeff2))
        
        # Constant term c^2 doesn't affect optimization (no variables), so we can ignore it