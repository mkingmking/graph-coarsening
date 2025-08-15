class Qubo:
    def __init__(self):
        self.dict = {}

    def add(self, key, value):
        if key in self.dict:
            self.dict[key] += value
        else:
            self.dict[key] = value

    def add_only_one_constraint(self, variables, penalty):
        for i in range(len(variables)):
            self.add((variables[i], variables[i]), -penalty)
            for j in range(i + 1, len(variables)):
                self.add((variables[i], variables[j]), 2 * penalty)
