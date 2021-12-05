from random import uniform, choice, randint, random
from functools import reduce

from numpy.lib.function_base import select
from models.ga_models.ga_base import GA_BaseModel
"""
    GA Model That Does Nothing.
    Used For Standard Training Of The Model.
"""


class GA_NoneModel:
    def __init__(self, config):
        pass
