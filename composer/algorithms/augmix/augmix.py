# Copyright 2021 MosaicML. All Rights Reserved.

from dataclasses import asdict, dataclass

import numpy as np
import torch
import yahp as hp
from PIL import Image
from PIL.Image import Image as ImageType

from composer.algorithms.algorithm_hparams import AlgorithmHparams
from composer.core.event import Event
from composer.core.types import Algorithm, Event, List, Logger, State
from composer.utils.augmentation_primitives import augmentation_sets
from composer.utils.data import add_dataset_transform


@dataclass
class AugMixHparams(AlgorithmHparams):

    severity: int = hp.optional(doc="Intensity of each augmentation. Ranges from 0 (none) to 10 (maximum)", default=3)
    depth: int = hp.optional(doc="Number of augmentations to compose in a row", default=-1)
    width: int = hp.optional(doc="Number of parallel augmentation chains to combine", default=3)
    alpha: float = hp.optional(doc="Mixing parameter for clean vs. augmented images.", default=1.0)
    augmentation_set: str = hp.optional(
        doc=
        "Set of augmentations to sample from. 'all', 'safe' (only augmentations that don't appear on CIFAR10C/ImageNet10C), or 'original'",
        default="all")

    def initialize_object(self) -> "AugMix":
        return AugMix(**asdict(self))


def augment_and_mix(img: ImageType = None,
                    severity: int = 3,
                    depth: int = -1,
                    width: int = 3,
                    alpha: float = 1.0,
                    augmentation_set: List = augmentation_sets["all"]) -> ImageType:
    """
    Perform augmentations.
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

    def __init__(self,
                 severity: int = 3,
                 depth: int = -1,
                 width: int = 3,
                 alpha: float = 1.0,
                 augmentation_set: str = "all"):
        """
        See documentation for RandAugment
        """
        super().__init__()
        if severity < 0 or severity > 10:
            raise ValueError("AugMix severity value must be 0 ≤ severity ≤ 10")
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

        return augment_and_mix(img=img,
                               severity=self.severity,
                               depth=self.depth,
                               width=self.width,
                               alpha=self.alpha,
                               augmentation_set=self.augmentation_set)


class AugMix(Algorithm):
    """
    Object that does AugMix (Hendrycks et al. (2020), AugMix: A Simple Data
        Processing Method to Improve Robustness and Uncertainty). Can be passed as a
        transform to torchvision.transforms.Compose().

    Args:
        severity: Severity of augmentation operators (between 1 to 10).
        width: Width of augmentation chains (number of parallel augmentations)
        depth: Depth of each augmentation chain. -1 enables stochastic depth
            uniformly from [1, 3]
        alpha: Probability coefficient for Beta and Dirichlet distributions. Sampling
            from Dirichlet determines relative weights of each augmented image
            sampling from beta determines relative weights of unaugmented and
            augmented images.
        augmentation_set: String, one of ["augmentations_all",
            "augmentations_corruption_safe", "augmentations_original"]. Set of
            augmentations to use. "augmentations_corruption_safe" excludes transforms
            that are part of the ImageNet-C/CIFAR10-C test sets.
            "augmentations_original" uses all augmentations, but some of the
            implementations are identical to the original github repo, which appears
            to contain implementation specificities for the augmentations "color",
            "contrast", "sharpness", and "brightness".
    """

    def __init__(self,
                 severity: int = 3,
                 depth: int = -1,
                 width: int = 3,
                 alpha: float = 1.0,
                 augmentation_set: str = "all"):
        if severity < 0 or severity > 10:
            raise ValueError("AugMix severity value must be 0 ≤ severity ≤ 10")
        if width < 1:
            raise ValueError("AugMix width must be ≥ 1")
        if augmentation_set not in augmentation_sets.keys():
            raise KeyError(f"AugMix augmentation_set is not one of {augmentation_sets.keys()}")
        self.hparams = AugMixHparams(severity=severity,
                                     depth=depth,
                                     width=width,
                                     alpha=alpha,
                                     augmentation_set=augmentation_set)

    def match(self, event: Event, state: State) -> bool:
        """ Runs on Event.TRAINING_START
        """
        return event == Event.TRAINING_START

    def apply(self, event: Event, state: State, logger: Logger) -> None:
        """ Inserts AugMix into the list of dataloader transforms
        """
        am = AugmentAndMixTransform(**self.hparams.to_dict())
        assert state.train_dataloader is not None, "Train Dataloader is not initialized."
        dataset = state.train_dataloader.dataset
        add_dataset_transform(dataset, am)
