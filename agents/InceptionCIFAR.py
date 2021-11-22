import numpy as np

from tqdm import tqdm
import itertools

import torch
from torch import nn

from torchvision.models import inception_v3
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from torch.optim import SGD
from torchvision import transforms
import torch.nn.functional as F

from agents.base import BaseAgent

from tensorboardX import SummaryWriter


class InceptionCIFAR(BaseAgent):

    def __init__(self, config):
        super().__init__(config)

        # define models
        self.model = inception_v3(progress=False)

        # define data_loader
        self.batch_size = config.batch_size
        transform = transforms.Compose([
            transforms.Resize(255),
            transforms.CenterCrop(299), 
            transforms.ToTensor(),
        ])
        cifar_data = CIFAR10('.', train=True, download=True, transform=transform)
        cifar_test_data = CIFAR10('.', train=False, download=True, transform=transform)
        # not yet support GPU
        self.data_loader = DataLoader(cifar_data, batch_size=config.batch_size, shuffle=True)
        self.test_loader = DataLoader(cifar_test_data, batch_size=config.batch_size)

        # define loss
        self.loss = nn.CrossEntropyLoss()

        # define optimizers for both generator and discriminator
        self.optimizer = SGD(self.model.parameters(), lr=config.learning_rate)

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
            print("Program will run on *****GPU-CUDA***** ")
        else:
            print("Program will run on *****CPU*****\n")

        # Model Loading from the latest checkpoint if not found start from scratch.
        self.load_checkpoint(self.config.checkpoint_file)
        # Summary Writer
        self.summary_writer = SummaryWriter()

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
        for epoch in range(self.config.current_epoch, self.config.max_epoch):
            self.current_epoch = epoch
            print(f"Epoch: {self.current_epoch + 1}")
            epoch_loss = self.train_one_epoch()
            test_loss = self.validate()
            self.summary_writer.add_scalar("training loss", epoch_loss / len(self.data_loader), self.current_epoch)
            self.summary_writer.add_scalar("test loss", test_loss / len(self.test_loader), self.current_epoch)

    def train_one_epoch(self):
        """
        One epoch of training
        :return:
        """
        self.model.train()
        epoch_loss = 0

        for x, y in self.data_loader:
            # x, y = x.to(self.device), y.to(self.device)

            y_pred, x = self.model(x)
            cur_loss = self.loss(y_pred, y)
            self.optimizer.zero_grad()
            cur_loss.backward()
            self.optimizer.step()
            epoch_loss += cur_loss.item()
            self.current_iteration += 1

        return epoch_loss

    def validate(self):
        """
        One cycle of model validation
        :return:
        """
        self.model.eval()
        test_loss = 0
        correct = 0

        for data, target in self.test_loader, 1:
            output = self.model(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()  # sum up batch loss
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(self.test_loader * self.batch_size)
        return test_loss

    def finalize(self):
        """
        Finalizes all the operations of the 2 Main classes of the process, the operator and the data loader
        :return:
        """
        pass