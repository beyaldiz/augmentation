import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, enable_cuda=True):
        super().__init__()
        self.enable_cuda = enable_cuda
        self.mlp = nn.Sequential(nn.Conv2d(3, 6, 5), nn.ReLU(),
                                 nn.MaxPool2d(2, 2), nn.Conv2d(6, 16, 5),
                                 nn.ReLU(), nn.MaxPool2d(2, 2), nn.Flatten(),
                                 nn.Linear(16 * 5 * 5, 120), nn.ReLU(),
                                 nn.Linear(120, 84), nn.ReLU(),
                                 nn.Linear(84, 10))

    def forward(self, x):
        if self.enable_cuda:
            return self.mlp(x.cuda())
        else:
            return self.mlp(x)
