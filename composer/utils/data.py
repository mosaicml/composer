# Copyright 2021 MosaicML. All Rights Reserved.

import collections.abc
import logging
import textwrap
from typing import List, Tuple, Union

import numpy as np
import torch
import torch.utils.data
from PIL import Image
from torchvision import transforms

from composer.core.types import Batch, Dataset, Tensor
from composer.utils.iter_helpers import ensure_tuple

log = logging.getLogger(__name__)


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


def add_dataset_transform(dataset, transform, location="end", pre_post: str = "pre") -> Dataset:
    """Flexibly add a transform to the dataset's collection of transforms. Warning: this
    function will not behave as expected if a dataset's collection of transforms contains
    nested torchvision.transforms.Compose (e.g dataset.transforms =
    transforms.Compose([transform0, transform1, transforms.Compose(...)])). This function
    should be updated to flatten nested compositions.

    Args:
        dataset: A torchvision-like dataset
        transform: Function to be added to the dataset's collection of transforms
        location (str, torchvision transform), optional: Where to insert the transform in
            the sequence of transforms. "end" will append to the end or "start" will
            insert at index 0, a transform (e.g. torchvision.transforms.ToTensor) will
            insert pre- or post- the first instance of `location`, determined by the
            `pre_post` argument. Default = "end". If `location` is not in list of
            transforms, will append to end.
        pre_post (str), optional: Whether to insert before ("pre") or after ("post") the
            transform string passed in 'location'. Ignored if transform = "start" or "end". Default = "pre".

    Returns:
        The original dataset. The transform is added in-place.
    """

    if not hasattr(dataset, "transform"):
        raise AttributeError(textwrap.dedent(f"""Dataset must have attribute `transform`."""))

    valid_non_transform_locations = ["start", "end"]
    valid_pre_post_values = ["pre", "post"]
    if location not in valid_non_transform_locations and pre_post not in valid_pre_post_values:
        raise ValueError(
            textwrap.dedent(
                f"Invalid value combination for argument `pre_post`: `{pre_post} and `location`: {location}. If location is not one of ['start', 'end'], `pre_post` must be one of ['pre', 'post']."
            ))

    added_transform = 0

    def added_transform_warning():
        log.warning(
            f"Transform {transform} added to dataset. Dataset now has the following transforms: {dataset.transform}")
        return 1

    if dataset.transform is None:
        dataset.transform = transform
        added_transform = added_transform_warning()
    # Check if dataset.transform is of type transforms. Compose and ensure idempotency by not adding transform if it's already present
    elif isinstance(dataset.transform,
                    transforms.Compose) and type(transform) not in [type(t) for t in dataset.transform.transforms]:
        if location == "end":
            dataset.transform.transforms.append(transform)
        elif location == "start":
            dataset.transform.transforms.insert(0, transform)
        else:
            insertion_index = len(dataset.transform.transforms)
            for i, t in enumerate(dataset.transform.transforms):
                if type(t) == location:
                    insertion_index = i
                    break
            if pre_post == "post":
                insertion_index += 1
            dataset.transform.transforms.insert(insertion_index, transform)
        added_transform = added_transform_warning()
    elif not isinstance(dataset.transform, transforms.Compose):
        # Transform is some other basic transform, join using Compose
        # Ensure idempotency by not adding transform if it's already present
        if not type(dataset.transform) == type(transform):
            if type(dataset.transform) == location and pre_post == "pre":
                dataset.transform = transforms.Compose([transform, dataset.transform])
            else:
                dataset.transform = transforms.Compose([dataset.transform, transform])
            added_transform = added_transform_warning()
    if not added_transform:
        log.warning(
            f"Unable to add transform {transform} added to dataset. Dataset has the following transforms: {dataset.transform}"
        )
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


def get_device_of_batch(batch: Batch) -> torch.device:
    """Returns the :class:`torch.device` of the batch.

    Args:
        batch (Batch): The batch to determine the device of.

    Returns:
        torch.device: The device that the batch is on.
    """
    if isinstance(batch, Tensor):
        return batch.device
    if isinstance(batch, (tuple, list)):  # BatchPair
        for sample in ensure_tuple(batch):
            for x in ensure_tuple(sample):
                for tensor in ensure_tuple(x):
                    return tensor.device

    if isinstance(batch, dict):  # BatchDict
        for x in batch.values():
            return x.device
    raise TypeError(f"Unsupported type for batch: {type(batch)}")
