from random import uniform, choice, randint, random
from functools import reduce

"""
Genetic algorithm model base class.
"""
class GABaseModel:
    def __init__(self, config):
        self.population_size = config.population_size
        self.augmentations = config.augmentations
        self.augmentation_ranges = config.augmentation_ranges
        self.genome_len = len(config.augmentations)
        self.crossover_rate = config.crossover_rate
        self.elitism_selection_rate = config.elitism_selection_rate
        self.populations = []
    
    def init_single_genome(self): # Initialize a single genome randomly
        return [uniform(*self.augmentation_ranges[i]) for i in range(self.genome_len)]

    def init_single_population(self): # Initialize a single population randomly
        return [self.init_single_genome() for _ in range(self.population_size)]
    
    def init_populations(self, dataset_len): # Initialize populations randomly
        self.dataset_len = dataset_len
        self.populations = [self.init_single_population() for _ in range(dataset_len)]

    def crossover(self, parent1, parent2): # Crossover two parents
        r = randint(0, self.genome_len)
        return parent1[:r] + parent2[r:]

    def mutate(self, genome, aug_idx): # Mutate one augmentation of a genome
        return genome[:aug_idx] + [uniform(*self.augmentation_ranges[aug_idx])] + genome[aug_idx + 1:]

    def is_valid(self, genome): # Check if a genome is valid
        return reduce(lambda x, y: x and y, [self.augmentation_ranges[i][0] <= genome[i] <= self.augmentation_ranges[i][1] for i in range(self.genome_len)])

    def generate_single_population(self, population): # Algorithm 3 in the paper
        children = []
        while len(children) < self.population_size:
            if random() < self.crossover_rate:
                parent1, parent2 = choice(population), choice(population)
                child = self.crossover(parent1, parent2)
            else:
                parent = choice(population)
                aug_idx = randint(0, self.genome_len - 1)
                child = self.mutate(parent, aug_idx)
            if self.is_valid(child):
                children.append(child)
        return children

    def generate_populations(self): # Algorithm 2, line 13 in the paper
        return [self.generate_single_population(self.populations[i]) for i in range(self.dataset_len)]
    
    def update_single_population(self, population, fitness): # Algorithm 2, line 16 in the paper
        indices = [i for i in range(self.population_size)]
        indices.sort(key=lambda x: fitness[x], reverse=True)
        choose_size = int(self.population_size * self.elitism_selection_rate)
        return [population[i] for i in indices[:choose_size]]

    def update_population(self, fitness, populations): # Algorithm 2, line 16 in the paper
        self.populations = [self.update_single_population(populations[i], fitness[i]) for i in range(self.dataset_len)]
        