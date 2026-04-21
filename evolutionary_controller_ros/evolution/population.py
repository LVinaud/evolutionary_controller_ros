"""Evolutionary operators: selection, crossover, mutation."""


class Population:
    def __init__(self, size):
        self.size = size
        self.individuals = []

    def select(self):
        raise NotImplementedError

    def crossover(self, parent_a, parent_b):
        raise NotImplementedError

    def mutate(self, genome):
        raise NotImplementedError
