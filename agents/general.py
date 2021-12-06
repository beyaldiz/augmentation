import numpy as np

from tqdm import tqdm
from itertools import product
from numpy import linspace
from random import shuffle

import torch
from torch import nn

from torchvision.datasets import *
from torch.utils.data import DataLoader
from torch.optim import SGD
from torch.optim import Adam
from torchvision import transforms
import torch.nn.functional as F

from agents.base import BaseAgent
from datasets.augmentable import AugmentableDataset

from utils.transformations import Transformations

from tensorboardX import SummaryWriter


class General(BaseAgent):
    def __init__(self, config):
        super().__init__(config)

        # define models
        self.model = self.import_model(config.dl_model)()
        self.ga_model = self.import_model(config.ga_model)(config)

        # define data_loader
        self.batch_size = config.batch_size
        pre_transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        # Transformations defined in config is loaded into Transformations class
        # Later it is used internally in AugmentableDataset to compose transformations using genomes
        self.transformations = Transformations(config)

        try:
            dataset = globals()[config.dataset]
            data = dataset('./data', train=True, download=True)
            test_data = dataset('./data', train=False, download=True)
        except:
            raise KeyError(
                "The dataset name is invalid, please visit https://pytorch.org/vision/stable/datasets.html"
            )

        # augmentation strategies: Random, W-10, SENSEI
        self.aug_dataset_train = AugmentableDataset(
            data.data,
            data.targets,
            self.transformations,
            pre_transform=pre_transform,
            shuffle=config.shuffle)

        self.aug_dataset_test = AugmentableDataset(test_data.data,
                                                   test_data.targets,
                                                   self.transformations,
                                                   pre_transform=pre_transform)

        # not yet support GPU
        self.data_loader = DataLoader(self.aug_dataset_train,
                                      batch_size=config.batch_size,
                                      shuffle=True)
        self.data_loader_ordered = DataLoader(self.aug_dataset_train,
                                              batch_size=config.batch_size,
                                              shuffle=False)
        self.test_loader = DataLoader(self.aug_dataset_test,
                                      batch_size=config.batch_size)

        # define loss
        self.loss = nn.CrossEntropyLoss()
        self.loss_single = nn.CrossEntropyLoss(reduction='none')

        # define optimizers for both generator and discriminator
        self.optimizer = Adam(self.model.parameters(), lr=config.learning_rate)

        # initialize counter
        self.current_epoch = 0
        self.current_iteration = 0
        self.best_metric = 0

        # set cuda flag
        self.is_cuda = torch.cuda.is_available()
        if self.is_cuda and not self.config.cuda:
            print(
                "WARNING: You have a CUDA device, so you should probably enable CUDA"
            )

        self.cuda = self.is_cuda & self.config.cuda
        self.model.enable_cuda = self.cuda

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
        print(f"\ntraining loss: {epoch_loss / len(self.data_loader)}")
        print(f"test loss: {test_loss / len(self.test_loader)}")
        print(f"test accuracy: {correct / len(self.test_loader.dataset)}")

        self.summary_writer.add_scalar("training loss",
                                       epoch_loss / len(self.data_loader),
                                       self.current_epoch)
        self.summary_writer.add_scalar("test loss",
                                       test_loss / len(self.test_loader),
                                       self.current_epoch)
        self.summary_writer.add_scalar("test accuracy",
                                       correct / len(self.test_loader.dataset),
                                       self.current_epoch)

        # visualize transformed images
        for transformed_images, _ in self.data_loader:
            break
        num_images = min(32, len(transformed_images))
        self.summary_writer.add_images("transformed images", transformed_images[:num_images], self.current_epoch)

        # robust accuracy
        if self.current_epoch % self.config.robust_interval == 0:
            robust_acc = self.robust_accuracy()
            self.summary_writer.add_scalar("robust accuracy",
                                           robust_acc,
                                           self.current_epoch)
            print(f"robust accuracy: {robust_acc}")
        
            if self.current_epoch == self.config.max_epoch - 1:
                print(f"final robust accuracy: {robust_acc}")
                self.summary_writer.add_scalar("final robust accuracy", robust_acc)

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
                if type(self.ga_model).__name__ != 'GA_NoneModel':
                    self.ga_model.init_populations(
                        len(self.data_loader.dataset))
                self.write_summary_per_epoch(epoch_loss, test_loss, correct)
                self.current_epoch = epoch
                continue

            if type(self.ga_model).__name__ != 'GA_NoneModel':
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
                inpt = x[i]
                y_pred = self.model(inpt)
                if self.cuda:
                    loss = self.loss_single(y_pred, y[i].cuda())
                else:
                    loss = self.loss_single(y_pred, y[i])
                batch_loss.append(loss.cpu().detach().numpy())
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
        # return 0
        self.model.train()
        self.aug_dataset_train.train_best()
        epoch_loss = 0

        for x, y in tqdm(self.data_loader):
            y_pred = self.model(x)
            if self.cuda:
                cur_loss = self.loss(y_pred, y.cuda())
            else:
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
        self.aug_dataset_test.train_best()
        test_loss = 0
        correct = 0

        for x, y in tqdm(self.test_loader):
            y_pred = self.model(x)
            if self.cuda:
                cur_loss = self.loss(y_pred, y.cuda())
            else:
                cur_loss = self.loss(y_pred, y)
            pred = y_pred.max(1)[1]
            correct += pred.cpu().eq(y).sum().item()
            test_loss += cur_loss.item()

        return test_loss, correct

    def robust_accuracy(self, num=3):
        # compute robust accuracy

        genomes = [list(linspace(*aug_ranges, num=num)) for aug_ranges in self.config.augmentation_ranges]
        genomes = product(*genomes)
        genomes = [list(genome) for genome in genomes]

        # Pick 5 random genomes
        # shuffle(genomes)
        # genomes = genomes[:5]

        print("\nComputing robust accuracy...")
        self.model.eval()
        self.aug_dataset_test.eval_children()
        children = [
            genomes for _ in range(len(self.test_loader.dataset))
        ]
        self.aug_dataset_test.update_children(children) 
        robusts, total = 0, 0
        for x, y in tqdm(self.test_loader):
            bs = len(x[0])
            res = torch.BoolTensor([True for _ in range(bs)])
            for i in range(len(x)):
                inpt = x[i]
                y_pred = self.model(inpt)
                pred = y_pred.max(1)[1]
                res = torch.logical_and(res, pred.cpu().eq(y[i]))
            res[res == True] = 1
            robusts += torch.sum(res)
            total += bs
        return robusts / total

    def finalize(self, num=3):
        """
        Finalizes all the operations of the 2 Main classes of the process, the operator and the data loader
        :return:
        """
        
        if self.config.max_epoch % self.config.robust_interval != 0:
            robust_acc = self.robust_accuracy()
            print(f"Robust accuracy: {robust_acc}")
            self.summary_writer.add_scalar("final robust accuracy", robust_acc)
