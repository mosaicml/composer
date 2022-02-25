# Copyright 2021 MosaicML. All Rights Reserved.

"""Core RandAugment code."""

import functools
import textwrap
import weakref
from typing import List, TypeVar

import numpy as np
import torch
from PIL.Image import Image as PillowImage
from torchvision.datasets import VisionDataset

from composer.algorithms.utils import augmentation_sets
from composer.algorithms.utils.augmentation_common import map_pillow_function
from composer.core.types import Algorithm, Event, Logger, State
from composer.datasets.utils import add_vision_dataset_transform

__all__ = ['RandAugment', "RandAugmentTransform", 'randaugment_image']

ImgT = TypeVar("ImgT", torch.Tensor, PillowImage)


def randaugment_image(img: ImgT,
                      severity: int = 9,
                      depth: int = 2,
                      augmentation_set: List = augmentation_sets["all"]) -> ImgT:
    """Randomly applies a sequence of image data augmentations
    (`Cubuk et al, 2019 <https://arxiv.org/abs/1909.13719>`_) to an image. See
    :class:`~composer.algorithms.randaugment.randaugment.RandAugment` or the :doc:`Method
    Card </method_cards/randaugment>` for details.

    Example:
        .. testcode::

            from composer.algorithms.randaugment import randaugment_image
            from composer.algorithms.utils import augmentation_sets
            randaugmented_image = randaugment_image(
                img=image,
                severity=9,
                depth=2,
                augmentation_set=augmentation_sets["all"]
            )

    Args:
        img (PIL.Image): Image or batch of images to be RandAugmented.
        severity (int, optional): See :class:`~composer.algorithms.randaugment.randaugment.RandAugment`.
        depth (int, optional): See :class:`~composer.algorithms.randaugment.randaugment.RandAugment`.
        augmentation_set (str, optional): See
            :class:`~composer.algorithms.randaugment.randaugment.RandAugment`.

    Returns:
        PIL.Image: RandAugmented image.
    """

    def _randaugment_pil_image(img: PillowImage, severity: int, depth: int, augmentation_set: List) -> PillowImage:
        # Iterate over augmentations
        for _ in range(depth):
            aug = np.random.choice(augmentation_set)
            img = aug(img, severity)
        return img

    f_pil = functools.partial(_randaugment_pil_image, severity=severity, depth=depth, augmentation_set=augmentation_set)
    return map_pillow_function(f_pil, img)


class RandAugmentTransform(torch.nn.Module):
    """Wraps :func:`~composer.algorithms.randaugment.randaugment.randaugment_image` in a
    ``torchvision``-compatible transform. See
    :class:`~composer.algorithms.randaugment.randaugment.RandAugment` or the :doc:`Method
    Card </method_cards/randaugment>` for more details.

    Example:
        .. testcode::

            import torchvision.transforms as transforms
            from composer.algorithms.randaugment import RandAugmentTransform
            randaugment_transform = RandAugmentTransform(
                severity=9,
                depth=2,
                augmentation_set="all"
            )
            composed = transforms.Compose([randaugment_transform, transforms.RandomHorizontalFlip()])
            transformed_image = composed(image)

    Args:
        severity (int, optional): See :class:`~composer.algorithms.randaugment.randaugment.RandAugment`.
        depth (int, optional): See :class:`~composer.algorithms.randaugment.randaugment.RandAugment`.
        augmentation_set (str, optional): See
            :class:`~composer.algorithms.randaugment.randaugment.RandAugment`.
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

    def forward(self, img: ImgT) -> ImgT:
        return randaugment_image(img=img,
                                 severity=self.severity,
                                 depth=self.depth,
                                 augmentation_set=self.augmentation_set)


class RandAugment(Algorithm):
    """Randomly applies a sequence of image data augmentations (`Cubuk et al, 2019 <https://arxiv.org/abs/1909.13719>`_)
    to an image.

    This algorithm runs on on :attr:`~composer.core.event.Event.INIT` to insert a dataset
    transformation. It is a no-op if this algorithm already applied itself on the
    :attr:`State.train_dataloader.dataset`.

    See the :doc:`Method Card </method_cards/randaugment>` for more details.

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
        augmentation_set (str, optional): Must be one of the following options:

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
                value bounded by 0.118 at a minimum, and a maximum value of
                :math:`intensity \\times 0.18 + .1`, which ranges from 0.28 (intensity =
                1) to 1.9 (intensity 10). These augmentations have different effects
                depending on whether they are < 0 or > 0 (or < 1 or > 1).
                "augmentations_all" uses implementations of "color", "contrast",
                "sharpness", and "brightness" that account for diverging effects around 0
                (or 1).

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
        ra = RandAugmentTransform(severity=self.severity, depth=self.depth, augmentation_set=self.augmentation_set)
        assert state.train_dataloader is not None
        dataset = state.train_dataloader.dataset
        if not isinstance(dataset, VisionDataset):
            raise TypeError(
                textwrap.dedent(f"""\
                To use {type(self).__name__}, the dataset must be a
                {VisionDataset.__qualname__}, not {type(dataset).__name__}"""))
        add_vision_dataset_transform(dataset, ra, is_tensor_transform=False)
        self._transformed_datasets.add(dataset)
