import torch
from torch.utils.data import Dataset

import torchvision.transforms as transforms


class AugmentableDataset(Dataset):
    def __init__(self, images, targets):
        """
        Args:
            images (Tensor): Tensor of shape (N, C, H, W)
            targets (Tensor): Tensor of shape (N, )
        """
        self.images = images
        self.targets = targets
        self.transformed_images = images.clone()
        self.transforms = [None for _ in range(self.images[0])]
    
    def __len__(self):
        return len(self.images.shape[0])
    
    def __getitem__(self, idx):
        return self.transformed_images[idx], self.targets[idx]
    
    def update_transform(self, idx, transform):
        self.transforms[idx] = transform
        self.transformed_images[idx] = transform(self.images[idx])
