import torch
import random
from torchvision import transforms
import torchvision.transforms.functional as F

def _rotate(angle):
    return lambda x: F.affine(x, angle=angle, translate=(0, 0), scale=1, shear=0)

def _translate_horizontal(percent):
    return lambda x: F.affine(x, angle=0, translate=(percent, 0), scale=1, shear=0)

def _translate_vertical(percent):
    return lambda x: F.affine(x, angle=0, translate=(0, percent), scale=1, shear=0)

def _shear(percent):
    return lambda x: F.affine(x, angle=0, translate=(0, 0), scale=1, shear=percent)

def _custom(arg):
    return lambda x: None

_transformations = {"rotate" : _rotate,
                    "translate_horizontal" : _translate_horizontal, 
                    "translate_vertical" : _translate_vertical, 
                    "shear" : _shear,
                    "custom": _custom}
    
class Transformations:
    def __init__(self, config):
        self.augmentations = config.augmentations
        self.genome_len = len(self.augmentations)
        self.transformations = [_transformations[augmentation] for augmentation in self.augmentations]
    
    def get_transformation(self, genome, shuffle=False):
        transformation_list = []
        for i in range(len(genome)):
            transformation = self.transformations[i]
            transformation_list.append(transformation(genome[i]))
        
        if shuffle:
            random.shuffle(transformation_list)

        return transforms.Compose(transformation_list)
