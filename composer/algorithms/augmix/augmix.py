# Copyright 2021 MosaicML. All Rights Reserved.

import textwrap
import weakref
from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np
import torch
import yahp as hp
from PIL import Image
from PIL.Image import Image as ImageType
from torchvision.datasets import VisionDataset

from composer.algorithms.algorithm_hparams import AlgorithmHparams
from composer.core.event import Event
from composer.core.types import Algorithm, Event, List, Logger, State
from composer.utils.augmentation_primitives import augmentation_sets
from composer.utils.data import add_dataset_transform


@dataclass
class AugMixHparams(AlgorithmHparams):
    """See :class:`AugMix`"""

    severity: int = hp.optional(doc="Intensity of each augmentation. Ranges from 0 (none) to 10 (maximum)", default=3)
    depth: int = hp.optional(doc="Number of augmentations to compose in a row", default=-1)
    width: int = hp.optional(doc="Number of parallel augmentation sequences to combine", default=3)
    alpha: float = hp.optional(doc="Mixing parameter for clean vs. augmented images.", default=1.0)
    augmentation_set: str = hp.optional(
        doc=
        "Set of augmentations to sample from. 'all', 'safe' (only augmentations that don't appear on CIFAR10C/ImageNet10C), or 'original'",
        default="all")

    def initialize_object(self) -> "AugMix":
        return AugMix(**asdict(self))


def augmix_image(img: Optional[ImageType] = None,
                 severity: int = 3,
                 depth: int = -1,
                 width: int = 3,
                 alpha: float = 1.0,
                 augmentation_set: List = augmentation_sets["all"]) -> ImageType:
    """Applies AugMix (`Hendrycks et al.

    <http://arxiv.org/abs/1912.02781>`_) data augmentation to an image. See :class:`AugMix` for details.
    """

    assert isinstance(img, ImageType) or isinstance(img, np.ndarray), "img must be a PIL.Image"
    chain_weights = np.float32(np.random.dirichlet([alpha] * width))
    mixing_weight = np.float32(np.random.beta(alpha, alpha))
    augmented_combination = np.zeros_like(img, dtype=np.float32)

    # Iterate over image chains
    for chain_i in range(width):
        augmented_image = img.copy()
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
    mixed = (1 - mixing_weight) * np.asarray(img) + mixing_weight * augmented_combination
    mixed = Image.fromarray(np.uint8(mixed))
    return mixed


class AugmentAndMixTransform(torch.nn.Module):
    """Wrapper module for :func:`augmix_image` that can be passed to :class:`torchvision.transforms.Compose`"""

    def __init__(self,
                 severity: int = 3,
                 depth: int = -1,
                 width: int = 3,
                 alpha: float = 1.0,
                 augmentation_set: str = "all"):
        super().__init__()
        if severity < 0 or severity > 10:
            raise ValueError("AugMix severity value must satisfy 0 ≤ severity ≤ 10")
        if width < 1:
            raise ValueError("AugMix width must be ≥ 1")
        if augmentation_set not in augmentation_sets.keys():
            raise KeyError(f"AugMix augmentation_set is not one of {augmentation_sets.keys()}")
        self.severity = severity
        self.depth = depth
        self.width = width
        self.alpha = alpha
        self.augmentation_set = augmentation_sets[augmentation_set]

    def forward(self, img: ImageType) -> ImageType:

        return augmix_image(img=img,
                            severity=self.severity,
                            depth=self.depth,
                            width=self.width,
                            alpha=self.alpha,
                            augmentation_set=self.augmentation_set)


class AugMix(Algorithm):
    """`AugMix <http://arxiv.org/abs/1912.02781>`_ creates ``width`` sequences of ``depth`` image augmentations, applies
    each sequence with random intensity, and returns a convex combination of the ``width`` augmented images and the
    original image.

    The coefficients for mixing the augmented images are drawn from a uniform
    ``Dirichlet(alpha, alpha, ...)`` distribution. The coefficient for mixing
    the combined augmented image and the original image is drawn from a
    ``Beta(alpha, alpha)`` distribution, using the same ``alpha``.

    This algorithm runs on on :attr:`Event.FIT_START` to insert a dataset transformation. It is a no-op if this algorithm already
    applied itself on the :attr:`State.train_dataloader.dataset`.

    Args:
        severity: severity of augmentations; ranges from 0
            (no augmentation) to 10 (most severe).
        width: number of augmentation sequences
        depth: number of augmentations per sequence. -1 enables stochastic depth
            sampled uniformly from [1, 3].
        alpha: pseudocount for Beta and Dirichlet distributions. Must be > 0.
            Higher values yield mixing coefficients closer to uniform
            weighting. As the value approaches 0, the mixing coefficients
            approach using only one version of each image.
        augmentation_set: must be one of the following options:

            * ``"augmentations_all"``
                Uses all augmentations from the paper.
            * ``"augmentations_corruption_safe"``
                Like ``"augmentations_all"``, but excludes transforms that are part of
                the ImageNet-C/CIFAR10-C test sets
            * ``"augmentations_original"``
                Like ``"augmentations_all"``, but some of the implementations
                are identical to the original Github repository, which contains
                implementation specificities for the augmentations
                ``"color"``, ``"contrast"``, ``"sharpness"``, and ``"brightness"``.
    """

    # TODO document each value of augmentation_set in more detail; i.e.,
    # which augmentations are actually used

    def __init__(self,
                 severity: int = 3,
                 depth: int = -1,
                 width: int = 3,
                 alpha: float = 1.0,
                 augmentation_set: str = "all"):
        if severity < 0 or severity > 10:
            raise ValueError("AugMix severity value must satisfy 0 ≤ severity ≤ 10")
        if width < 1:
            raise ValueError("AugMix width must be ≥ 1")
        if augmentation_set not in augmentation_sets.keys():
            raise KeyError(f"AugMix augmentation_set is not one of {augmentation_sets.keys()}")
        self.severity = severity
        self.depth = depth
        self.width = width
        self.alpha = alpha
        self.augmentation_set = augmentation_set
        self._transformed_datasets = weakref.WeakSet()

    def match(self, event: Event, state: State) -> bool:
        return event == Event.FIT_START and state.train_dataloader.dataset not in self._transformed_datasets

    def apply(self, event: Event, state: State, logger: Logger) -> None:
        """Inserts AugMix into the list of dataloader transforms."""
        am = AugmentAndMixTransform(severity=self.severity,
                                    depth=self.depth,
                                    width=self.width,
                                    alpha=self.alpha,
                                    augmentation_set=self.augmentation_set)
        dataset = state.train_dataloader.dataset
        if not isinstance(dataset, VisionDataset):
            raise TypeError(
                textwrap.dedent(f"""\
                To use {type(self).__name__}, the dataset must be a
                {VisionDataset.__qualname__}, not {type(dataset).__name__}"""))
        add_dataset_transform(dataset, am, is_tensor_transform=False)
        self._transformed_datasets.add(dataset)
