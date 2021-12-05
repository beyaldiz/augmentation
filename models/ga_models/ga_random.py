from random import uniform, choice, randint, random
from functools import reduce

from numpy.lib.function_base import select
from models.ga_models.ga_base import GA_BaseModel
"""
    Random Augmentation Algorithm In the Paper When The Population Size is 1
    W-10 Algorithm In the Paper When The Population Size is 10 
"""


class GA_RandomModel(GA_BaseModel):
    def generate_populations(self):
        return [self.init_single_population() for _ in range(self.dataset_len)]

    def update_population(self, fitness, populations):
        pass
