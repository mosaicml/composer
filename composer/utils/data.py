# Copyright 2021 MosaicML. All Rights Reserved.

import collections.abc

import torch.utils.data
from torchvision import transforms

from composer.core.types import Dataset


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
