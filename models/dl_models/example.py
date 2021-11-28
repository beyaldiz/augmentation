import torch
import torch.nn as nn


class ExampleModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
