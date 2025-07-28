from itertools import product

class Qubo:
    def __init__(self):
        self.dict = dict()

    def create_field(self, field):
        if field not in self.dict:
            self.dict[field] = 0

    def add_only_one_constraint(self, variables, const):
        # This function correctly implements the penalty C * ( (sum(vars) - 1)^2 ).
        # This expands to C * (sum(v_i^2) - 2 * sum(v_i) + 2 * sum(v_i * v_j) + 1).
        # Since v_i^2 = v_i for binary variables, it simplifies to
        # C * (-sum(v_i) + 2 * sum(v_i * v_j) + 1).
        # We ignore the constant offset +1.
        
        # Add the linear terms: -C * v_i
        for var in variables:
            self.create_field((var, var))
            self.dict[(var, var)] -= const

        # Add the quadratic terms: +2C * v_i * v_j for i != j
        # The itertools.product covers all pairs, so we use 2 * C.
        for field in product(variables, variables):
            if field[0] == field[1]:
                continue
            self.create_field(field)
            self.dict[field] += 2 * const # Corrected to 2 * const

    def add(self, field, value):
        self.create_field(field)
        self.dict[field] += value

    def merge_with(self, qubo, const1, const2):
        for field in self.dict:
            self.dict[field] *= const1
        for field in qubo.dict:
            self.create_field(field)
            self.dict[field] += qubo.dict[field] * const2

    def get_dict(self):
        # Return a copy to avoid external modifications
        return self.dict.copy()

