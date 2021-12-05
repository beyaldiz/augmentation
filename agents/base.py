"""
The Base Agent class, where all other agents inherit from, that contains definitions for all the necessary functions
"""
import logging
from models.dl_models import *
from models.ga_models import *


class BaseAgent:
    """
    This base class will contain the base functions to be overloaded by any agent you will implement.
    """
    def __init__(self, config):
        self.config = config

    def load_checkpoint(self, file_name):
        """
        Latest checkpoint loader
        :param file_name: name of the checkpoint file
        :return:
        """
        raise NotImplementedError

    def save_checkpoint(self, file_name="checkpoint.pth.tar", is_best=0):
        """
        Checkpoint saver
        :param file_name: name of the checkpoint file
        :param is_best: boolean flag to indicate whether current checkpoint's metric is the best so far
        :return:
        """
        raise NotImplementedError

    def run(self):
        """
        The main operator
        :return:
        """
        raise NotImplementedError

    def train(self):
        """
        Main training loop
        :return:
        """
        raise NotImplementedError

    def train_one_epoch(self):
        """
        One epoch of training
        :return:
        """
        raise NotImplementedError

    def validate(self):
        """
        One cycle of model validation
        :return:
        """
        raise NotImplementedError

    def finalize(self):
        """
        Finalizes all the operations of the 2 Main classes of the process, the operator and the data loader
        :return:
        """
        raise NotImplementedError

    def import_model(self, model):
        """
        Imports a model from a file
        :param model: the model to be imported
        :return:
        """
        return globals()[model]
