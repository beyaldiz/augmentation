from random import uniform, choice, randint, random
from functools import reduce

from numpy.lib.function_base import select
from models.ga_models.ga_base import GA_BaseModel
"""
Genetic algorithm with Roulette-Wheel Selection.
"""


class GA_RWModel(GA_BaseModel):
    def roulette_wheel_select(self, population, fitness):
        # Select ONE chromosome using Roulette-Wheel method
        f_sum = sum(fitness)
        pick = random() * f_sum
        tmp = 0
        for i in range(self.population_size):
            tmp += fitness[i]
            if tmp > pick:
                return population[i]
        return population[self.population_size - 1]

    def update_single_population(self, population, fitness):
        # Redefine selection with Roulette-Wheel method
        new_population = []
        for i in range(self.population_size):
            chosen = self.roulette_wheel_select(population, fitness)
            new_population.append(chosen)

        return new_population
