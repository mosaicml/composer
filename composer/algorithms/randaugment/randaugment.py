# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Core RandAugment code."""

import functools
import textwrap
import weakref
from typing import TypeVar

import numpy as np
import torch
import torch.utils.data
from PIL.Image import Image as PillowImage
from torchvision.datasets import VisionDataset

from composer.algorithms.utils import augmentation_sets
from composer.algorithms.utils.augmentation_common import map_pillow_function
from composer.core import Algorithm, Event, State
from composer.loggers import Logger
from composer.utils import add_vision_dataset_transform

__all__ = ['RandAugment', 'RandAugmentTransform', 'randaugment_image']

ImgT = TypeVar('ImgT', torch.Tensor, PillowImage)


def randaugment_image(
    img: ImgT,
    severity: int = 9,
    depth: int = 2,
    augmentation_set: list = augmentation_sets['all'],
) -> ImgT:
    """Randomly applies a sequence of image data augmentations  to an image or batch of images.

    This technique is adapted from `Cubuk et al, 2019 <https://arxiv.org/abs/1909.13719>`_).

    See :class:`.RandAugment` or the :doc:`Method Card </method_cards/randaugment>`
    for details. This function only acts on a single image (or batch of images) per call and
    is unlikely to be used in a training loop. Use :class:`.RandAugmentTransform` to use
    :class:`.RandAugment` as part of a :class:`torchvision.datasets.VisionDataset` ``transform``.

    Example:
        .. testcode::

            import composer.functional as cf

            from composer.algorithms.utils import augmentation_sets

            randaugmented_image = cf.randaugment_image(
                img=image,
                severity=9,
                depth=2,
                augmentation_set=augmentation_sets["all"]
            )

    Args:
        img (PIL.Image.Image | torch.Tensor): Image or batch of images to be RandAugmented.
        severity (int, optional): See :class:`.RandAugment`.
        depth (int, optional): See :class:`.RandAugment`.
        augmentation_set (str, optional): See :class:`.RandAugment`.

    Returns:
        PIL.Image: RandAugmented image.
    """

    def _randaugment_pil_image(img: PillowImage, severity: int, depth: int, augmentation_set: list) -> PillowImage:
        # Iterate over augmentations
        for _ in range(depth):
            aug = np.random.choice(augmentation_set)
            img = aug(img, severity)
        return img

    f_pil = functools.partial(_randaugment_pil_image, severity=severity, depth=depth, augmentation_set=augmentation_set)
    return map_pillow_function(f_pil, img)


class RandAugmentTransform(torch.nn.Module):
    """Wraps :func:`.randaugment_image` in a ``torchvision``-compatible transform.

    See :class:`.RandAugment` or the :doc:`Method Card </method_cards/randaugment>` for more details.

    Example:
        .. testcode::

            import torchvision.transforms as transforms
            from composer.algorithms.randaugment import RandAugmentTransform

            randaugment_transform = RandAugmentTransform(
                severity=9,
                depth=2,
                augmentation_set="all"
            )
            composed = transforms.Compose([
                randaugment_transform,
                transforms.RandomHorizontalFlip()
            ])
            transformed_image = composed(image)

    Args:
        severity (int, optional): See :class:`.RandAugment`.
        depth (int, optional): See :class:`.RandAugment`.
        augmentation_set (str, optional): See
            :class:`.RandAugment`.
    """

    def __init__(self, severity: int = 9, depth: int = 2, augmentation_set: str = 'all'):
        super().__init__()
        if severity < 0 or severity > 10:
            raise ValueError('RandAugment severity value must satisfy 0 ≤ severity ≤ 10')
        if depth < 0:
            raise ValueError('RandAugment depth value must be ≥ 0')
        if augmentation_set not in augmentation_sets.keys():
            raise KeyError(f'RandAugment augmentation_set is not one of {augmentation_sets.keys()}')
        self.severity = severity
        self.depth = depth
        self.augmentation_set = augmentation_sets[augmentation_set]

    def forward(self, img: ImgT) -> ImgT:
        return randaugment_image(
            img=img,
            severity=self.severity,
            depth=self.depth,
            augmentation_set=self.augmentation_set,
        )


class RandAugment(Algorithm):
    """Randomly applies a sequence of image data augmentations to an image.

    This algorithm (`Cubuk et al, 2019 <https://arxiv.org/abs/1909.13719>`_) runs on
    :attr:`.Event.INIT` to insert a dataset
    transformation. It is a no-op if this algorithm already applied itself on the
    :attr:`.State.train_dataloader.dataset`.

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
            in the original paper. Default: ``9``.
        depth (int, optional): Depth of augmentation chain. N in the original paper.
            Default: ``2``.
        augmentation_set (str, optional): Must be one of the following options
            as also described in :attr:`.augmentation_primitives.augmentation_sets`:

            * ``"all"``
                Uses all augmentations from the paper.
            * ``"safe"``
                Like ``"all"``, but excludes transforms that are part of
                the ImageNet-C/CIFAR10-C test sets
            * ``"original"``
                Like ``"all"``, but some of the implementations
                are identical to the original Github repository, which contains
                implementation specificities for the augmentations
                ``"color"``, ``"contrast"``, ``"sharpness"``, and ``"brightness"``. The
                original implementations have an intensity sampling scheme that samples a
                value bounded by 0.118 at a minimum, and a maximum value of
                :math:`intensity \\times 0.18 + .1`, which ranges from 0.28 (intensity =
                1) to 1.9 (intensity 10). These augmentations have different effects
                depending on whether they are < 0 or > 0 (or < 1 or > 1).
                ``"all"`` uses implementations of ``"color"``, ``"contrast"``,
                ``"sharpness"``, and ``"brightness"`` that account for diverging effects
                around 0 (or 1).

            Default: ``"all"``.
    """

    def __init__(self, severity: int = 9, depth: int = 2, augmentation_set: str = 'all'):
        if severity < 0 or severity > 10:
            raise ValueError('RandAugment severity value must be 0 ≤ severity ≤ 10')
        if augmentation_set not in augmentation_sets.keys():
            raise KeyError(f'randaugment_augmentation_set is not one of {augmentation_sets.keys()}')
        self.severity = severity
        self.depth = depth
        self.augmentation_set = augmentation_set
        self._transformed_datasets = weakref.WeakSet()

    def match(self, event: Event, state: State) -> bool:
        if event != Event.FIT_START:
            return False
        assert state.dataloader is not None, 'dataloader should be defined on fit start'
        if not isinstance(state.dataloader, torch.utils.data.DataLoader):
            raise TypeError(f'{type(self).__name__} requires a PyTorch dataloader.')
        return state.dataloader.dataset not in self._transformed_datasets

    def apply(self, event: Event, state: State, logger: Logger) -> None:
        ra = RandAugmentTransform(severity=self.severity, depth=self.depth, augmentation_set=self.augmentation_set)
        assert isinstance(state.dataloader, torch.utils.data.DataLoader), 'The dataloader type is checked on match()'
        dataset = state.dataloader.dataset
        if not isinstance(dataset, VisionDataset):
            raise TypeError(
                textwrap.dedent(
                    f"""\
                To use {type(self).__name__}, the dataset must be a
                {VisionDataset.__qualname__}, not {type(dataset).__name__}""",
                ),
            )
        add_vision_dataset_transform(dataset, ra, is_tensor_transform=False)
        self._transformed_datasets.add(dataset)
