# Copyright 2021 MosaicML. All Rights Reserved.

"""Core RandAugment code."""

import textwrap
import weakref
from typing import Optional

import numpy as np
import torch
from PIL.Image import Image as ImageType
from torchvision.datasets import VisionDataset

from composer.core.types import Algorithm, Event, List, Logger, State
from composer.utils.augmentation_primitives import augmentation_sets
from composer.utils.data import add_dataset_transform

__all__ = ['RandAugment', "RandAugmentTransform", 'randaugment_image']


def randaugment_image(img: Optional[ImageType] = None,
                      severity: int = 9,
                      depth: int = 2,
                      augmentation_set: List = augmentation_sets["all"]) -> ImageType:
    """Randomly applies a sequence of image data augmentations (`Cubuk et al. 2019
    <https://openaccess.thecvf.com/content_CVPRW_2020/papers/w40/Cubuk_Randaugment_Practical_Automated_Data_Augmentation_With_a_Reduced_Search_Space_CVPRW_2020_paper.pdf>`_)
    to an image. See :class:`~composer.algorithms.randaugment.randaugment.RandAugment` for
    details.

    Example:
        .. testcode::

            from composer.algorithms.randaugment import randaugment_image
            from composer.utils.augmentation_primitives import augmentation_sets
            randaugmented_image = randaugment_image(
                img=image,
                severity=9,
                depth=2,
                augmentation_set=augmentation_sets["all"]
            )
    """

    # Iterate over augmentations
    for _ in range(depth):
        aug = np.random.choice(augmentation_set)
        img = aug(img, severity)
    assert img is not None
    return img


class RandAugmentTransform(torch.nn.Module):
    """Wraps :func:`~composer.algorithms.randaugment.randaugment.randaugment_image` in a
    ``torchvision``-compatible transform. 

    Example:
        .. testcode::

            import torchvision.transforms
            from composer.algorithms.randaugment import RandAugmentTransform 
            randaugment_transform = RandAugmentTransform(
                severity=9,
                depth=2,
                augmentation_set="all"
            )
            composed = transforms.Compose([randaugment_transform, transforms.RandomHorizontalFlip()])
    """

    def __init__(self, severity: int = 9, depth: int = 2, augmentation_set: str = "all"):
        super().__init__()
        if severity < 0 or severity > 10:
            raise ValueError("RandAugment severity value must satisfy 0 ≤ severity ≤ 10")
        if depth < 0:
            raise ValueError("RandAugment depth value must be ≥ 0")
        if augmentation_set not in augmentation_sets.keys():
            raise KeyError(f"RandAugment augmentation_set is not one of {augmentation_sets.keys()}")
        self.severity = severity
        self.depth = depth
        self.augmentation_set = augmentation_sets[augmentation_set]

    def forward(self, img: ImageType) -> ImageType:

        return randaugment_image(img=img,
                                 severity=self.severity,
                                 depth=self.depth,
                                 augmentation_set=self.augmentation_set)


class RandAugment(Algorithm):
    """Randomly apply a sequence of image data augmentations (`Cubuk et al. 2019
    <https://openaccess.thecvf.com/content_CVPRW_2020/papers/w40/Cubuk_Randaugment_Practical_Automated_Data_Augmentation_With_a_Reduced_Search_Space_CVPRW_2020_paper.pdf>`_).

    This algorithm runs on on :attr:`Event.INIT` to insert a dataset transformation. It is a no-op if this algorithm already
    applied itself on the :attr:`State.train_dataloader.dataset`.

    Example:
        .. testcode::

            from composer.algorithms import RandAugment
            from composer.trainer import Trainer
            randaugment_algorithm = RandAugment(
                severity=9,
                depth=2,
                augmentation_set="all"
            )
            trainer = Trainer(
                model=model,
                train_dataloader=train_dataloader,
                eval_dataloader=eval_dataloader,
                max_duration="1ep",
                algorithms=[randaugment_algorithm],
                optimizers=[optimizer]
            )

    Args:
        severity (int, optional): Severity of augmentation operators (between 1 to 10). M
            in the original paper. Default = ``9``.
        depth (int, optional): Depth of augmentation chain. N in the original paper
            Default = ``2``.
        augmentation_set (str, optional): must be one of the following options:

            * ``"augmentations_all"``
                Uses all augmentations from the paper.
            * ``"augmentations_corruption_safe"``
                Like ``"augmentations_all"``, but excludes transforms that are part of
                the ImageNet-C/CIFAR10-C test sets
            * ``"augmentations_original"``
                Like ``"augmentations_all"``, but some of the implementations
                are identical to the original Github repository, which contains
                implementation specificities for the augmentations
                ``"color"``, ``"contrast"``, ``"sharpness"``, and ``"brightness"``. The
                original implementations have an intensity sampling scheme that samples a
                value bounded by 0.118 at a minimum, and a maximum value of intensity*0.18
                + .1, which ranges from 0.28 (intensity = 1) to 1.9 (intensity 10). These
                augmentations have different effects depending on whether they are < 0 or
                > 0 (or < 1 or > 1). "augmentations_all" uses implementations of "color",
                "contrast", "sharpness", and "brightness" that account for diverging
                effects around 0 (or 1).

            Default = ``"all"``.
    """

    def __init__(self, severity: int = 9, depth: int = 2, augmentation_set: str = "all"):
        if severity < 0 or severity > 10:
            raise ValueError("RandAugment severity value must be 0 ≤ severity ≤ 10")
        if augmentation_set not in augmentation_sets.keys():
            raise KeyError(f"randaugment_augmentation_set is not one of {augmentation_sets.keys()}")
        self.severity = severity
        self.depth = depth
        self.augmentation_set = augmentation_set
        self._transformed_datasets = weakref.WeakSet()

    def match(self, event: Event, state: State) -> bool:
        return event == Event.FIT_START and state.train_dataloader.dataset not in self._transformed_datasets

    def apply(self, event: Event, state: State, logger: Logger) -> None:
        """Inserts RandAugment into the list of dataloader transforms.

        Args:
            event (Event): the current event
            state (State): the current trainer state
            logger (Logger): the training logger
        """
        ra = RandAugmentTransform(severity=self.severity, depth=self.depth, augmentation_set=self.augmentation_set)
        assert state.train_dataloader is not None
        dataset = state.train_dataloader.dataset
        if not isinstance(dataset, VisionDataset):
            raise TypeError(
                textwrap.dedent(f"""\
                To use {type(self).__name__}, the dataset must be a
                {VisionDataset.__qualname__}, not {type(dataset).__name__}"""))
        add_dataset_transform(dataset, ra, is_tensor_transform=False)
        self._transformed_datasets.add(dataset)
