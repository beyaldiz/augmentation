import numpy as np

from tqdm import tqdm
from itertools import product
from numpy import linspace
import json

import torch
from torch import nn

from torchvision.datasets import *
from torch.utils.data import DataLoader
from torch.optim import *
from torch.optim.lr_scheduler import StepLR
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

        # Pre-transformations
        pre_transform_train = transforms.Compose([
            transforms.ToTensor(),            
            transforms.RandomHorizontalFlip(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        pre_transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
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
            pre_transform=pre_transform_train,
            shuffle=config.shuffle)

        self.aug_dataset_test = AugmentableDataset(test_data.data,
                                                   test_data.targets,
                                                   self.transformations,
                                                   pre_transform=pre_transform_test)

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

        # Hard-coded for experiments
        if config.optimizer == "SGD":
            self.optimizer = SGD(self.model.parameters(), lr=config.learning_rate, momentum=0.9, weight_decay=config.weight_decay, nesterov=True)
        elif config.optimizer == "Adam":
            self.optimizer = Adam(self.model.parameters(), lr=config.learning_rate)
        
        # define optimizers for both generator and discriminator
        else:
            try:
                optimizer = globals()[config.optimizer]
            except:
                raise KeyError(
                    "The optimizer name is invalid, please visit https://pytorch.org/docs/stable/optim.html"
                )
            self.optimizer = optimizer(self.model.parameters(), lr=config.learning_rate)
        
        # Hard coded for experiments
        self.scheduler = StepLR(self.optimizer, step_size=80, gamma=0.1)

        # initialize counter
        self.current_epoch = 0
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

    def write_summary_per_epoch(self, epoch_loss, test_loss, epoch_correct, test_correct):
        
        # log accuracy and loss values
        summaries = {
            "training loss": epoch_loss / len(self.data_loader),
            "test loss": test_loss / len(self.test_loader),
            "training accuracy": epoch_correct / len(self.data_loader.dataset),
            "test accuracy": test_correct / len(self.test_loader.dataset),
        }

        print("")
        for key, value in summaries.items():
            print(f"{key}: {value}")
            self.summary_writer.add_scalar(key, value, self.current_epoch)

        # visualize transformed images
        for transformed_images, _ in self.data_loader_ordered:
            break
        num_images = min(self.config.num_log_samples, len(transformed_images))
        self.summary_writer.add_images("transformed images", transformed_images[:num_images], self.current_epoch)

        # log GA of transformations
        def pretty_json(hp):
            json_hp = json.dumps(hp, indent=2)
            return "".join("\t" + line for line in json_hp.splitlines(True)) 

        if type(self.ga_model).__name__ != 'GA_NoneModel' and self.current_epoch > 0:
            text = dict()
            for i in range(self.config.num_log_samples):
                text[i] = dict(zip(self.config.augmentations, self.aug_dataset_train.best_genomes[i]))
            self.summary_writer.add_text("transformations", pretty_json(text), self.current_epoch)

        # robust accuracy
        if (self.current_epoch + 1) % self.config.robust_interval == 0:
            robust_acc = self.robust_accuracy()
            self.summary_writer.add_scalar("robust accuracy",
                                           robust_acc,
                                           self.current_epoch)
            print(f"robust accuracy: {robust_acc}")
        
            # if self.current_epoch + 1 == self.config.max_epoch:
            #     print(f"final robust accuracy: {robust_acc}")
            #     self.summary_writer.add_scalar("final robust accuracy", robust_acc)

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
                epoch_loss, epoch_correct = self.train_one_epoch()
                test_loss, test_correct = self.validate()
                if type(self.ga_model).__name__ != 'GA_NoneModel':
                    self.ga_model.init_populations(
                        len(self.data_loader.dataset))
                self.write_summary_per_epoch(epoch_loss, test_loss, epoch_correct, test_correct)
                self.current_epoch = epoch
                continue

            if type(self.ga_model).__name__ != 'GA_NoneModel':
                self.genetic_evolve_one_epoch()
            epoch_loss, epoch_correct = self.train_one_epoch()
            test_loss, test_correct = self.validate()
            self.write_summary_per_epoch(epoch_loss, test_loss, epoch_correct, test_correct)

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
        with torch.no_grad():
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
        self.model.train()
        self.aug_dataset_train.train_best()
        epoch_loss, epoch_correct = 0, 0

        for x, y in tqdm(self.data_loader):
            y_pred = self.model(x)
            if self.cuda:
                cur_loss = self.loss(y_pred, y.cuda())
            else:
                cur_loss = self.loss(y_pred, y)
            self.optimizer.zero_grad()
            cur_loss.backward()
            self.optimizer.step()
            pred = y_pred.max(1)[1]
            epoch_correct += pred.cpu().eq(y).sum().item()
            epoch_loss += cur_loss.item()

        self.scheduler.step()
        return epoch_loss, epoch_correct

    def validate(self):
        """
        One cycle of model validation
        :return:
        """
        print("Model validation...")
        self.model.eval()
        self.aug_dataset_test.train_best()
        test_loss, test_correct = 0, 0

        with torch.no_grad():
            for x, y in tqdm(self.test_loader):
                y_pred = self.model(x)
                if self.cuda:
                    cur_loss = self.loss(y_pred, y.cuda())
                else:
                    cur_loss = self.loss(y_pred, y)
                pred = y_pred.max(1)[1]
                test_correct += pred.cpu().eq(y).sum().item()
                test_loss += cur_loss.item()

        return test_loss, test_correct

    def robust_accuracy(self, num=3):
        # compute robust accuracy

        genomes = [list(linspace(*aug_ranges, num=num)) for aug_ranges in self.config.augmentation_ranges]
        genomes = product(*genomes)
        genomes = [list(genome) for genome in genomes]

        print("\nComputing robust accuracy...")
        self.model.eval()
        self.aug_dataset_test.eval_children()
        children = [
            genomes for _ in range(len(self.test_loader.dataset))
        ]
        self.aug_dataset_test.update_children(children) 
        robusts, total = 0, 0
        with torch.no_grad():
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
        pass
        # if self.config.max_epoch % self.config.robust_interval != 0:
        #     robust_acc = self.robust_accuracy()
        #     print(f"Robust accuracy: {robust_acc}")
        #     self.summary_writer.add_scalar("final robust accuracy", robust_acc)
