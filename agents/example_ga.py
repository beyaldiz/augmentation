import numpy as np

from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import DataLoader

from agents.base import BaseAgent
from datasets.augmentable import AugmentableDataset
from models.ga_models.example import ExampleModel

from tensorboardX import SummaryWriter


class ExampleAgentGA(BaseAgent):

    def __init__(self, config):
        super().__init__(config)

        # define models
        self.model = None

        # define data_loader
        self.train_dataset = AugmentableDataset() # update this
        self.train_steps = len(self.train_dataset) // config.batch_size
        self.data_loader_ga = None
        self.data_loader_dl = DataLoader(self.train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)

        # define genetic model
        self.genetic_model = None

        # define loss
        self.loss = None

        # define optimizers for both generator and discriminator
        self.optimizer = None

        # initialize counter
        self.current_epoch = 0
        self.current_iteration = 0
        self.best_metric = 0

        # set cuda flag
        self.is_cuda = torch.cuda.is_available()
        if self.is_cuda and not self.config.cuda:
            print("WARNING: You have a CUDA device, so you should probably enable CUDA")

        self.cuda = self.is_cuda & self.config.cuda

        # set the manual seed for torch
        self.manual_seed = self.config.seed
        if self.cuda:
            torch.cuda.manual_seed_all(self.manual_seed)
            torch.cuda.set_device(self.config.gpu_device)
            self.model = self.model.cuda()
            self.loss = self.loss.cuda()
            self.device = torch.device("cuda")
            print("Program will run on *****GPU-CUDA***** ")
        else:
            self.device = torch.device("cpu")
            print("Program will run on *****CPU*****\n")

        # Model Loading from the latest checkpoint if not found start from scratch.
        self.load_checkpoint(self.config.checkpoint_file)
        # Summary Writer
        self.summary_writer = None

    def load_checkpoint(self, file_name):
        """
        Latest checkpoint loader
        :param file_name: name of the checkpoint file
        :return:
        """
        pass

    def save_checkpoint(self, file_name="checkpoint.pth.tar", is_best=0):
        """
        Checkpoint saver
        :param file_name: name of the checkpoint file
        :param is_best: boolean flag to indicate whether current checkpoint's accuracy is the best so far
        :return:
        """
        pass

    def run(self):
        """
        The main operator
        :return:
        """
        try:
            self.train()

        except KeyboardInterrupt:
            print("You have entered CTRL+C.. Wait to finalize")

    def train(self):
        """
        Main training loop
        :return:
        """
        self.train_one_epoch()
        self.genetic_model.initialize_population()
        for epoch in range(self.config.current_epoch, self.config.max_epoch):
            self.current_epoch = epoch
            self.ga_selection_one_epoch()
            self.train_one_epoch()

    def ga_selection_one_epoch(self):
        """
        One epoch of genetic algorithm selection
        :return:
        """
        pass

    def train_one_epoch(self):
        """
        One epoch of training
        :return:
        """
        tqdm_batch = tqdm(self.data_loader_dl, total=self.train_steps)
        self.model.train()
        epoch_loss = 0

        for batch_idx, (x, y) in enumerate(tqdm_batch):
            x, y = x.to(self.device), y.to(self.device)
            
            y_pred = self.model(x)
            cur_loss = self.loss(y_pred, y)

            self.optimizer.zero_grad()
            cur_loss.backward()
            self.optimizer.step()

            epoch_loss += cur_loss.item()
            self.current_iteration += 1

    def validate(self):
        """
        One cycle of model validation
        :return:
        """
        pass

    def finalize(self):
        """
        Finalizes all the operations of the 2 Main classes of the process, the operator and the data loader
        :return:
        """
        pass