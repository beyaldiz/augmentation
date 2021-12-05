import torch
import torch.nn as nn


class ResNet101(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = torch.hub.load('pytorch/vision:v0.10.0',
                                    'resnet101',
                                    pretrained=True)

    def forward(self, x):
        return self.model(x)
