import random


class GAModel:
    def __init__(self, config):
        self.population_size = config.population_size
        self.augmentations = config.augmentations
        self.augmentation_ranges = config.augmentation_ranges
        self.genome_len = len(config.augmentations)
        self.populations = []

    def init_single_genome(self):
        genome = []
        for i in range(self.genome_len):
            genome.append(
                random.uniform(self.augmentation_ranges[i][0],
                               self.augmentation_ranges[i][1]))
        return genome

    def init_single_population(self):
        single_population = []
        for i in range(self.population_size):
            single_population.append(self.init_single_genome())
        return single_population

    def init_populations(self, dataset_len):
        self.dataset_len = dataset_len
        self.populations = []
        for i in range(dataset_len):
            self.populations.append(self.init_single_population())

    def generate_populations(self):
        self.init_populations(self.dataset_len)
        return self.populations

    def update_population(self, population, fitness):
        self.generate_populations()
        return
