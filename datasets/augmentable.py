import torch
from torch.utils.data import Dataset

import torchvision.transforms as transforms


class AugmentableDataset(Dataset):
    def __init__(self,
                 images,
                 targets,
                 transformations=None,
                 pre_transform=None,
                 post_transform=None,
                 shuffle=None):
        """
        Images and targets are passed as any type.
        Pre transform is applied in the beginning of getitem.
        Transformations passed in self.transforms applied in the middle (i.e augmentations from the genomes).
        Post transform is applied in the end of getitem.

        Args:
            images (Any)
            targets (Any)
            transformations (Transformations)
        """
        self.images = images
        self.targets = targets
        self.transformations = transformations
        self.pre_transform = pre_transform
        self.post_transform = post_transform
        self.shuffle = shuffle
        self.transforms = [None for _ in range(self.images.shape[0])]
        self.best_genomes = [None for _ in range(self.images.shape[0])]
        self._eval_children = False
        self._children = None

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        if self._eval_children:
            instances = []
            for genome in self._children[idx]:
                tr_array = []
                if self.pre_transform != None:
                    tr_array.append(self.pre_transform)
                transform = self.transformations.get_transformation(
                    genome, self.shuffle)
                tr_array.append(transform)
                if self.post_transform != None:
                    tr_array.append(self.post_transform)

                # this might be vulnerable (treating any TypeError as image type error)
                # and assume that transform.toTensor is the first element of the array
                try:
                    tr = transforms.Compose(tr_array)
                    instances.append(tr(self.images[idx]))
                except TypeError as e:
                    if self.pre_transform:
                        tr_array.pop(0)
                        tr = transforms.Compose(tr_array)
                        instances.append(tr(self.images[idx]))
                    else:
                        raise e
            return instances, [
                self.targets[idx] for _ in range(len(instances))
            ]

        else:
            tr_array = []
            if self.pre_transform != None:
                tr_array.append(self.pre_transform)
            if self.transforms[idx] != None:
                tr_array.append(self.transforms[idx])
            if self.post_transform != None:
                tr_array.append(self.post_transform)
            if len(tr_array) != 0:
                # this might be vulnerable (treating any TypeError as image type error)
                # and assume that transform.toTensor is the first element of the array
                if self.pre_transform:
                    try:
                        tr = transforms.Compose(tr_array)
                        return tr(self.images[idx]), self.targets[idx]
                    except TypeError as e:
                        if self.pre_transform:
                            tr_array.pop(0)
                            tr = transforms.Compose(tr_array)
                            return tr(self.images[idx]), self.targets[idx]
                        else:
                            raise e
            else:
                return self.images[idx], self.targets[idx]

    def pick_best_child(self, f_best):
        """
        f_best is a list of indices that correspond to the children that have the largest loss

        Args:
        f_best (numpy.array, shape: (N, )): indices of the children that have the largest loss 
        """
        for i in range(f_best.shape[0]):
            self.best_genomes[i] = self._children[i][f_best[i]]
            self.transforms[i] = self.transformations.get_transformation(
                self._children[i][f_best[i]])

    def eval_children(self):
        """
        Switching to evaluation of children's mode
        """
        self._eval_children = True

    def train_best(self):
        """
        Switching to training with the best children mode
        """
        self._eval_children = False

    def update_children(self, children):
        """
        Updates the children array
        """
        self._children = children
