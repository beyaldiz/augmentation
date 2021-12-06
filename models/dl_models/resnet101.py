import torch
import torch.nn as nn


class ResNet101(nn.Module):
    def __init__(self, cuda=True):
        super().__init__()
        self.cuda = cuda
        self.model = torch.hub.load('pytorch/vision:v0.10.0',
                                    'resnet101',
                                    pretrained=False)

    def forward(self, x):
        if self.cuda:
            return self.model(x.cuda())
        else:
            return self.model(x)
