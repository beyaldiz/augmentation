from random import uniform, choice, choices, randint, random
from functools import reduce

from numpy.lib.function_base import select
from models.ga_models.ga_base import GA_BaseModel
"""
Genetic algorithm with Tournament Selection.
"""


class GA_Tournament(GA_BaseModel):
    def tournament_select(self, population, fitness):
        # Select ONE chromosome using Tournament method
        indices = [i for i in range(self.population_size)]
        candidates = choices(indices, k=5)  # Randomly select 5 candidates
        candidates.sort(key=lambda x: fitness[x], reverse=True)

        return population[candidates[0]]

    def update_single_population(self, population, fitness):
        # Redefine selection with Tournament method
        new_population = []
        for i in range(self.population_size):
            chosen = self.tournament_select(population, fitness)
            new_population.append(chosen)

        return new_population
