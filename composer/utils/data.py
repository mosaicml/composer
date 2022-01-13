# Copyright 2021 MosaicML. All Rights Reserved.

import collections.abc
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.utils.data
from PIL import Image
from torchvision import transforms

from composer.core.types import Batch, Dataset, Tensor


class NormalizationFn:
    """Normalizes input data and removes the background class from target data if desired.

    Args:
        mean (Tuple[float, float, float]): the mean pixel value for each channel (RGB) for the dataset.
        std (Tuple[float, float, float]): the standard deviation pixel value for each channel (RGB) for the dataset.
        ignore_background (bool): if true, ignore the background class in the training loss. Only used in semantic
            segmentation. Default is False.

    """

    def __init__(self, 
                 mean: Tuple[float, float, float], 
                 std: Tuple[float, float, float], 
                 ignore_background: bool = False):
        self.mean = mean
        self.std = std
        self.ignore_background = ignore_background

    def __call__(self, batch: Batch):
        xs, ys = batch
        assert isinstance(xs, Tensor)
        assert isinstance(ys, Tensor)
        device = xs.device

        if not isinstance(self.mean, torch.Tensor):
            self.mean = torch.tensor(self.mean, device=device)
            self.mean = self.mean.view(1, 3, 1, 1)
        if not isinstance(self.std, torch.Tensor):
            self.std = torch.tensor(self.std, device=device)
            self.std = self.std.view(1, 3, 1, 1)

        xs = xs.float()
        xs = xs.sub_(self.mean).div_(self.std)
        if self.ignore_background:
            ys = ys.sub_(1)
        return xs, ys


def pil_image_collate(batch: List[Tuple[Image.Image, Union[Image.Image, Tensor]]],
                      memory_format: torch.memory_format = torch.contiguous_format) -> Tuple[Tensor, Tensor]:
    """Constructs a torch tensor batch for training from samples in PIL image format.

    Args:
        batch (List[Tuple[Image.Image, Union[Image.Image, torch.Tensor]]]): list of input-target pairs to be separated
            and aggregated into batches.
        memory_format (torch.memory_format): the memory format for the input and target tensors.

    Returns:
        image_tensor (torch.Tensor): torch tensor containing a batch of images.
        target_tensor (torch.Tensor): torch tensor containing a batch of targets.

    """
    imgs = [sample[0] for sample in batch]
    w, h = imgs[0].size
    image_tensor = torch.zeros((len(imgs), 3, h, w), dtype=torch.uint8).contiguous(memory_format=memory_format)

    # Check if the targets are images
    targets = [sample[1] for sample in batch]
    if isinstance(targets[0], Image.Image):
        target_dims = (len(targets), targets[0].size[1], targets[0].size[0])  # type: ignore
    else:
        target_dims = (len(targets),)
    target_tensor = torch.zeros(target_dims, dtype=torch.int64).contiguous(memory_format=memory_format)

    for i, img in enumerate(imgs):
        nump_array = np.asarray(img, dtype=np.uint8)
        if nump_array.ndim < 3:
            nump_array = np.expand_dims(nump_array, axis=-1)

        nump_array = np.rollaxis(nump_array, 2).copy()
        if nump_array.shape[0] != 3:
            assert nump_array.shape[0] == 1, "unexpected shape"
            nump_array = np.resize(nump_array, (3, h, w))
        assert image_tensor.shape[1:] == nump_array.shape, "shape mismatch"

        image_tensor[i] += torch.from_numpy(nump_array)
        target_tensor[i] += torch.from_numpy(np.array(targets[i], dtype=np.int64))

    return image_tensor, target_tensor


def add_dataset_transform(dataset, transform):
    """Flexibly add a transform to the dataset's collection of transforms.

    Args:
        dataset: A torchvision-like dataset
        transform: Function to be added to the dataset's collection of transforms

    Returns:
        The original dataset. The transform is added in-place.
    """

    if not hasattr(dataset, "transform"):
        raise ValueError(f"Dataset of type {type(dataset)} has no attribute 'transform'. Expected TorchVision dataset.")

    if dataset.transform is None:
        dataset.transform = transform
    elif hasattr(dataset.transform, "transforms"):  # transform is a Compose
        dataset.transform.transforms.append(transform)
    else:  # transform is some other basic transform, join using Compose
        dataset.transform = transforms.Compose([dataset.transform, transform])

    return dataset


def get_subset_dataset(size: int, dataset: Dataset):
    """Returns a subset dataset

    Args:
        size (int): Maximum szie of the dataset
        dataset (Dataset): The dataset to subset

    Raises:
        ValueError: If the ``size`` is greater than ``len(dataset)``

    Returns:
        Dataset: The subset dataset
    """
    if isinstance(dataset, collections.abc.Sized) and len(dataset) < size:
        raise ValueError(f"The dataset length ({len(dataset)}) is less than the requested size ({size}).")
    dataset = torch.utils.data.Subset(dataset, list(range(size)))
    return dataset
