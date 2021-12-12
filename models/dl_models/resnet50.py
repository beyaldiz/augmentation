import torch
import torch.nn as nn


class ResNet50(nn.Module):
    def __init__(self, enable_cuda=True):
        super().__init__()
        self.enable_cuda = enable_cuda
        self.model = torch.hub.load('pytorch/vision:v0.10.0',
                                    'resnet50',
                                    pretrained=False)

    def forward(self, x):
        if self.enable_cuda:
            return self.model(x.cuda())
        else:
            return self.model(x)
