import torch
import random
from torchvision import transforms
import torchvision.transforms.functional as F
"""
    Rotate by given angle
"""


def _rotate(angle):
    return lambda x: F.affine(
        x, angle=angle, translate=(0, 0), scale=1, shear=0)


"""
    Translate horizontally by given factor
"""


def _translate_horizontal(factor):
    return lambda x: F.affine(
        x, angle=0, translate=(factor, 0), scale=1, shear=0)


"""
    Translate vertically by given factor
"""


def _translate_vertical(factor):
    return lambda x: F.affine(
        x, angle=0, translate=(0, factor), scale=1, shear=0)


"""
    Shear by given factor
"""


def _shear(factor):
    return lambda x: F.affine(
        x, angle=0, translate=(0, 0), scale=1, shear=factor)


"""
    Zoom by given factor
"""


def _zoom(factor):
    return lambda x: F.affine(
        x, angle=0, translate=(0, 0), scale=factor, shear=0)


"""
    Increase the pixel values by delta
"""


def _brightness(delta):
    return lambda x: x + delta


"""
    Provide contrast by multiplying the pixel values by a factor
"""


def _contrast(factor):
    return lambda x: x * factor


def _custom(arg):
    return lambda x: None


_transformations = {
    "rotate": _rotate,
    "translate_horizontal": _translate_horizontal,
    "translate_vertical": _translate_vertical,
    "shear": _shear,
    "brightness": _brightness,
    "contrast": _contrast,
    "zoom": _zoom,
    "custom": _custom
}


class Transformations:
    def __init__(self, config):
        self.augmentations = config.augmentations
        self.genome_len = len(self.augmentations)
        self.transformations = [
            _transformations[augmentation]
            for augmentation in self.augmentations
        ]

    def get_transformation(self, genome, shuffle=False):
        transformation_list = []
        for i in range(len(genome)):
            transformation = self.transformations[i]
            transformation_list.append(transformation(genome[i]))

        if shuffle:
            random.shuffle(transformation_list)

        return transforms.Compose(transformation_list)
