from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch
import yahp as hp
from torchvision import transforms
from torchvision.datasets import cityscapes


class RandomResize(torch.nn.Module):

    def __init__(self, min_resize_factor, max_resize_factor, factor_step_size, interpolation):
        super().__init__()
        self.resize_factors = np.arange(min_resize_factor, max_resize_factor + factor_step_size, factor_step_size)
        self.interpolation = interpolation

    def forward(self, x):
        resize_factor = np.random.choice(self.resize_factors)
        new_dims = (x.width * resize_factor, x.height * resize_factor)
        return transforms.functional.resize(x, new_dims, interpolation=self.interpolation)


from composer.datasets.hparams import DataloaderSpec, DatasetHparams
"""
Needed functionality:
- Be able to either train on fine annotations only or both fine and coarse
"""


def random_resize(img):
    transform = transforms.Resize(np.random.choice(np.arange(0.5, 2.25, 0.25)))
    return transform(img)


@dataclass
class CityscapesDatasetHparams(DatasetHparams):

    #resize_size: int = hp.required("resize size") # might not need this, all images are the same size?
    crop_size: int = hp.required("crop size")  # train is 769 for Deeplabv3-ResNet101
    is_train: bool = hp.required("whether to load the training or validation dataset")
    datadir: str = hp.required("data directory")
    min_resize_factor: float = hp.optional("Minimum factor to resize the image by", default=0.5)
    max_resize_factor: float = hp.optional("Maximum factor to resize the image by", default=2.0)
    factor_step_size: float = hp.optional("Step size for scale factors in between the minimum and the maximum",
                                          default=0.25)
    drop_last: bool = hp.optional("Whether to drop the last samples for the last batch", default=True)
    shuffle: bool = hp.optional("Whether to shuffle the dataset for each epoch", default=True)

    def initialize_object(self) -> DataloaderSpec:
        datadir = self.datadir
        is_train = self.is_train

        if is_train:
            train_img_transforms: List[torch.nn.Module] = []
            train_target_transformers: List[torch.nn.Module] = []
            train_img_transforms.append(
                RandomResize(self.min_resize_factor,
                             self.max_resize_factor,
                             self.factor_step_size,
                             interpolation=transforms.InterpolationMode.BILINEAR))
            train_target_transforms.append(
                RandomResize(self.min_resize_factor,
                             self.max_resize_factor,
                             self.factor_step_size,
                             interpolation=transforms.InterpolationMode.NEAREST))
