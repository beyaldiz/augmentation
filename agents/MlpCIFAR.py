import numpy as np

from tqdm import tqdm
import itertools

import torch
from torch import nn

from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from torch.optim import SGD
from torchvision import transforms
import torch.nn.functional as F

import matplotlib.pyplot as plt

from agents.base import BaseAgent
from datasets.augmentable import AugmentableDataset
from models.ga_models.ga_model import GAModel
from models.dl_models.mlp import MLP
from utils.transformations import Transformations

from tensorboardX import SummaryWriter

class MlpCIFAR(BaseAgent):

    def __init__(self, config):
        super().__init__(config)

        # define models
        self.model = MLP()
        self.ga_model = GAModel(config)

        # define data_loader
        self.batch_size = config.batch_size
        pre_transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        # Transformations defined in config is loaded into Transformations class
        # Later it is used internally in AugmentableDataset to compose transformations using genomes
        self.transformations = Transformations(config)

        cifar_data = CIFAR10('./data', train=True, download=True)
        cifar_test_data = CIFAR10('./data', train=False, download=True, transform=pre_transform)

        self.aug_dataset_train = AugmentableDataset(cifar_data.data[::4], cifar_data.targets[::4], self.transformations, pre_transform=pre_transform)

        # not yet support GPU
        self.data_loader = DataLoader(self.aug_dataset_train, batch_size=config.batch_size, shuffle=True)
        self.data_loader_ordered = DataLoader(self.aug_dataset_train, batch_size=config.batch_size, shuffle=False)
        self.test_loader = DataLoader(cifar_test_data, batch_size=config.batch_size)

        # define loss
        self.loss = nn.CrossEntropyLoss()
        self.loss_single = nn.CrossEntropyLoss(reduction='none')

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
        self.summary_writer = SummaryWriter(self.config.summary_dir)

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
    
    def write_summary_per_epoch(self, epoch_loss, test_loss, correct):
        self.summary_writer.add_scalar("training loss", epoch_loss / len(self.data_loader), self.current_epoch)
        self.summary_writer.add_scalar("test loss", test_loss / len(self.test_loader), self.current_epoch)
        self.summary_writer.add_scalar("test accuracy", correct / len(self.test_loader.dataset), self.current_epoch)

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
            print(f"\nEpoch: {self.current_epoch + 1}")
            if epoch == 0:
                epoch_loss = self.train_one_epoch()
                test_loss, correct = self.validate()
                self.ga_model.init_populations(len(self.data_loader.dataset))
                self.write_summary_per_epoch(epoch_loss, test_loss, correct)
                self.current_epoch = epoch
                continue
            
            self.genetic_evolve_one_epoch()
            epoch_loss = self.train_one_epoch()
            test_loss, correct = self.validate()
            self.write_summary_per_epoch(epoch_loss, test_loss, correct)
    
    def genetic_evolve_one_epoch(self):
        """
        One epoch of genetic evolution
        :return:
        """
        print("Genetic evolution...")
        self.model.eval()
        self.aug_dataset_train.eval_children()
        children = self.ga_model.generate_populations()
        self.aug_dataset_train.update_children(children)

        f_array = []
        for x, y in tqdm(self.data_loader_ordered):
            batch_loss = []
            for i in range(len(x)):
                inpt = x[i].flatten(start_dim=1)
                y_pred = self.model(inpt)
                loss = self.loss_single(y_pred, y[i])
                batch_loss.append(loss.detach().numpy())
            f_array.append(np.stack(batch_loss))
        f = np.concatenate(f_array, axis=1).transpose()
        f_best = f.argmax(axis=1)
        self.aug_dataset_train.pick_best_child(f_best)
        self.ga_model.update_population(f, children)
        

    def train_one_epoch(self):
        """
        One epoch of training
        :return:
        """
        print("Model training...")
        self.model.train()
        self.aug_dataset_train.train_best()
        epoch_loss = 0

        for x, y in tqdm(self.data_loader):
            x = x.flatten(start_dim=1)
            y_pred = self.model(x)
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
        print("Model validation...")
        self.model.eval()
        test_loss = 0
        correct = 0

        for x, y in tqdm(self.test_loader):
            x = x.flatten(start_dim=1)
            y_pred = self.model(x)
            cur_loss = self.loss(y_pred, y)
            pred = y_pred.max(1)[1]
            correct += pred.eq(y).sum().item()
            test_loss += cur_loss.item()

        return test_loss, correct

    def finalize(self):
        """
        Finalizes all the operations of the 2 Main classes of the process, the operator and the data loader
        :return:
        """
        pass