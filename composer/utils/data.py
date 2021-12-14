# Copyright 2021 MosaicML. All Rights Reserved.

import collections.abc
import textwrap
from typing import List, Sequence

import torch.utils.data
from torch.functional import Tensor
from torchvision import transforms

from composer.core.types import Batch, Dataset


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


def default_batch_split_fn(batch: Batch, n_microbatches: int) -> List[Batch]:
    if not isinstance(batch, Sequence):
        raise ValueError(f'split_fn requires batch be a tuple pair of tensors, got {type(batch)}')
    x, y = batch
    if isinstance(x, Tensor) and isinstance(y, Tensor):
        return list(zip(x.chunk(n_microbatches), y.chunk(n_microbatches)))
    if isinstance(x, List) and isinstance(y, List):
        return list(
            zip(
                [x[i::n_microbatches] for i in range(n_microbatches)],
                [y[i::n_microbatches] for i in range(n_microbatches)],
            ))
    raise ValueError(
        textwrap.dedent("""The default_batch_split_fn is unable to split the output of the dataloader.
        Please define a split_fn in your dataloader spec."""))
