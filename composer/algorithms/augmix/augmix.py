# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Core AugMix classes and functions."""

import functools
import textwrap
import weakref
from typing import List, TypeVar

import numpy as np
import torch
import torch.utils.data
from PIL import Image
from PIL.Image import Image as PillowImage
from torchvision.datasets import VisionDataset

from composer.algorithms.utils import augmentation_sets
from composer.algorithms.utils.augmentation_common import map_pillow_function
from composer.core import Algorithm, Event, State
from composer.datasets.utils import add_vision_dataset_transform
from composer.loggers import Logger

__all__ = ['AugMix', 'AugmentAndMixTransform', 'augmix_image']

ImgT = TypeVar('ImgT', torch.Tensor, PillowImage)


def augmix_image(img: ImgT,
                 severity: int = 3,
                 depth: int = -1,
                 width: int = 3,
                 alpha: float = 1.0,
                 augmentation_set: List = augmentation_sets['all']) -> ImgT:
    r"""Applies the AugMix (`Hendrycks et al, 2020 <http://arxiv.org/abs/1912.02781>`_) data augmentation.

    This function works on a single image or batch of images. See :class:`.AugMix` and
    the :doc:`Method Card </method_cards/augmix>` for details. This function only acts on a
    single image (or batch) per call and is unlikely to be used in a training loop.
    Use :class:`.AugmentAndMixTransform` to use AugMix as
    part of a :class:`torchvision.datasets.VisionDataset`\'s ``transform``.

    Example:
        .. testcode::

            import composer.functional as cf

            from composer.algorithms.utils import augmentation_sets

            augmixed_image = cf.augmix_image(
                img=image,
                severity=3,
                width=3,
                depth=-1,
                alpha=1.0,
                augmentation_set=augmentation_sets["all"]
            )

    Args:
        img (PIL.Image.Image | torch.Tensor): Image or batch of images to be AugMix'd.
        severity (int, optional): See :class:`.AugMix`.
        depth (int, optional): See :class:`.AugMix`.
        width (int, optional): See :class:`.AugMix`.
        alpha (float, optional): See :class:`.AugMix`.
        augmentation_set (str, optional): See
            :class:`.AugMix`.

    Returns:
         PIL.Image: AugMix'd image.
    """

    def _augmix_pil_image(img_pil: PillowImage, severity: int, depth: int, width: int, alpha: float,
                          augmentation_set: List) -> PillowImage:
        chain_weights = np.random.dirichlet([alpha] * width).astype(np.float32)
        mixing_weight = np.float32(np.random.beta(alpha, alpha))
        augmented_combination = np.zeros_like(img_pil, dtype=np.float32)

        # Iterate over image chains
        for chain_i in range(width):
            augmented_image = img_pil.copy()
            # Determine depth of current augmentation chain
            if depth > 0:
                d = depth
            else:
                d = np.random.randint(1, 4)
            # Iterate through chain depth
            for _ in range(d):
                aug = np.random.choice(augmentation_set)
                augmented_image = aug(augmented_image, severity)
            augmented_combination += chain_weights[chain_i] * np.asarray(augmented_image)
        mixed = (1 - mixing_weight) * np.asarray(img_pil) + mixing_weight * augmented_combination
        mixed = Image.fromarray(np.uint8(mixed))
        return mixed

    f_pil = functools.partial(_augmix_pil_image,
                              severity=severity,
                              depth=depth,
                              width=width,
                              alpha=alpha,
                              augmentation_set=augmentation_set)
    return map_pillow_function(f_pil, img)


class AugmentAndMixTransform(torch.nn.Module):
    """Wrapper module for :func:`.augmix_image` that can
    be passed to :class:`torchvision.transforms.Compose`. See
    :class:`.AugMix` and the :doc:`Method Card
    </method_cards/augmix>` for details.

    Example:
        .. testcode::

            import torchvision.transforms as transforms

            from composer.algorithms.augmix import AugmentAndMixTransform

            augmix_transform = AugmentAndMixTransform(
                severity=3,
                width=3,
                depth=-1,
                alpha=1.0,
                augmentation_set="all"
            )
            composed = transforms.Compose([
                augmix_transform,
                transforms.RandomHorizontalFlip()
            ])
            transformed_image = composed(image)

    Args:
        severity (int, optional): See :class:`.AugMix`.
        depth (int, optional): See :class:`.AugMix`.
        width (int, optional): See :class:`.AugMix`.
        alpha (float, optional): See :class:`.AugMix`.
        augmentation_set (str, optional): See
            :class:`.AugMix`.
    """

    def __init__(self,
                 severity: int = 3,
                 depth: int = -1,
                 width: int = 3,
                 alpha: float = 1.0,
                 augmentation_set: str = 'all'):
        super().__init__()
        if severity < 0 or severity > 10:
            raise ValueError('AugMix severity value must satisfy 0 ≤ severity ≤ 10')
        if width < 1:
            raise ValueError('AugMix width must be ≥ 1')
        if augmentation_set not in augmentation_sets.keys():
            raise KeyError(f'AugMix augmentation_set is not one of {augmentation_sets.keys()}')
        self.severity = severity
        self.depth = depth
        self.width = width
        self.alpha = alpha
        self.augmentation_set = augmentation_sets[augmentation_set]

    def forward(self, img: PillowImage) -> PillowImage:

        return augmix_image(img=img,
                            severity=self.severity,
                            depth=self.depth,
                            width=self.width,
                            alpha=self.alpha,
                            augmentation_set=self.augmentation_set)


class AugMix(Algorithm):
    r"""The AugMix data augmentation technique.

    AugMix (`Hendrycks et al, 2020 <http://arxiv.org/abs/1912.02781>`_) creates ``width`` sequences of ``depth``
    image augmentations, applies each sequence with random intensity, and returns a convex combination of the ``width``
    augmented images and the original image.  The coefficients for mixing the augmented images are drawn from a uniform
    ``Dirichlet(alpha, alpha, ...)`` distribution. The coefficient for mixing the combined augmented image and the
    original image is drawn from a ``Beta(alpha, alpha)`` distribution, using the same ``alpha``.

    This algorithm runs on on :attr:`.Event.FIT_START` to insert a dataset transformation.
    It is a no-op if this algorithm already applied itself on the :attr:`State.train_dataloader.dataset`.

    See the :doc:`Method Card </method_cards/augmix>` for more details.

    Example:
        .. testcode::

            from composer.algorithms import AugMix
            from composer.trainer import Trainer

            augmix_algorithm = AugMix(
                severity=3,
                width=3,
                depth=-1,
                alpha=1.0,
                augmentation_set="all"
            )
            trainer = Trainer(
                model=model,
                train_dataloader=train_dataloader,
                eval_dataloader=eval_dataloader,
                max_duration="1ep",
                algorithms=[augmix_algorithm],
                optimizers=[optimizer]
            )

    Args:
        severity (int, optional): Severity of augmentations; ranges from 0
            (no augmentation) to 10 (most severe). Default: ``3``.
        depth (int, optional): Number of augmentations per sequence. -1 enables stochastic
            depth sampled uniformly from ``[1, 3]``. Default: ``-1``.
        width (int, optional): Number of augmentation sequences. Default: ``3``.
        alpha (float, optional): Pseudocount for Beta and Dirichlet distributions. Must be
            > 0.  Higher values yield mixing coefficients closer to uniform weighting. As
            the value approaches 0, the mixing coefficients approach using only one
            version of each image. Default: ``1.0``.
        augmentation_set (str, optional): Must be one of the following options as also described
            in :attr:`~composer.algorithms.utils.augmentation_primitives.augmentation_sets`:

            * ``"all"``
                Uses all augmentations from the paper.
            * ``"safe"``
                Like ``"all"``, but excludes transforms that are part of
                the ImageNet-C/CIFAR10-C test sets.
            * ``"original"``
                Like ``"all"``, but some of the implementations
                are identical to the original Github repository, which contains
                implementation specificities for the augmentations
                ``"color"``, ``"contrast"``, ``"sharpness"``, and ``"brightness"``. The
                original implementations have an intensity sampling scheme that samples a
                value bounded by 0.118 at a minimum, and a maximum value of
                :math:`intensity \times 0.18 + .1`, which ranges from 0.28 (intensity = 1)
                to 1.9 (intensity 10). These augmentations have different effects
                depending on whether they are < 0 or > 0 (or < 1 or > 1).
                ``"all"`` uses implementations of ``"color"``, ``"contrast"``,
                ``"sharpness"``, and ``"brightness"`` that account for diverging effects around 0
                (or 1).

            Default: ``"all"``.
    """

    # TODO document each value of augmentation_set in more detail; i.e.,
    # which augmentations are actually used

    def __init__(self,
                 severity: int = 3,
                 depth: int = -1,
                 width: int = 3,
                 alpha: float = 1.0,
                 augmentation_set: str = 'all'):
        if severity < 0 or severity > 10:
            raise ValueError('AugMix severity value must satisfy 0 ≤ severity ≤ 10')
        if width < 1:
            raise ValueError('AugMix width must be ≥ 1')
        if augmentation_set not in augmentation_sets.keys():
            raise KeyError(f'AugMix augmentation_set is not one of {augmentation_sets.keys()}')
        self.severity = severity
        self.depth = depth
        self.width = width
        self.alpha = alpha
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
        am = AugmentAndMixTransform(severity=self.severity,
                                    depth=self.depth,
                                    width=self.width,
                                    alpha=self.alpha,
                                    augmentation_set=self.augmentation_set)
        assert isinstance(state.dataloader, torch.utils.data.DataLoader), 'dataloader type checked on match()'
        dataset = state.dataloader.dataset
        if not isinstance(dataset, VisionDataset):
            raise TypeError(
                textwrap.dedent(f"""\
                To use {type(self).__name__}, the dataset must be a
                {VisionDataset.__qualname__}, not {type(dataset).__name__}"""))
        add_vision_dataset_transform(dataset, am, is_tensor_transform=False)
        self._transformed_datasets.add(dataset)
