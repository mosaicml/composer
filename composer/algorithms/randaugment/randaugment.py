# Copyright 2021 MosaicML. All Rights Reserved.

from dataclasses import asdict, dataclass

import numpy as np
import torch
import yahp as hp
from PIL.Image import Image as ImageType

from composer.algorithms.algorithm_hparams import AlgorithmHparams
from composer.core.types import Algorithm, Event, List, Logger, State
from composer.utils.augmentation_primitives import augmentation_sets
from composer.utils.data import add_dataset_transform


@dataclass
class RandAugmentHparams(AlgorithmHparams):

    severity: int = hp.optional(doc="Intensity of each augmentation. Ranges from 0 (none) to 10 (maximum)", default=9)
    depth: int = hp.optional(doc="Number of augmentations to compose in a row", default=2)
    augmentation_set: str = hp.optional(
        doc=
        "Set of augmentations to sample from. 'all', 'safe' (only augmentations that don't appear on CIFAR10C/ImageNet10C), or 'original'",
        default="all")

    def initialize_object(self) -> "RandAugment":
        return RandAugment(**asdict(self))


def randaugment(img: ImageType = None,
                severity: int = 9,
                depth: int = 2,
                augmentation_set: List = augmentation_sets["all"]) -> ImageType:
    """
    Perform augmentations.
    """

    # Iterate over augmentations
    for _ in range(depth):
        aug = np.random.choice(augmentation_set)
        img = aug(img, severity)
    assert img is not None
    return img


class RandAugmentTransform(torch.nn.Module):

    def __init__(self, severity: int = 9, depth: int = 2, augmentation_set: str = "all"):
        """
        See documentation for RandAugment
        """
        super().__init__()
        if severity < 0 or severity > 10:
            raise ValueError("RandAugment severity value must be 0 ≤ severity ≤ 10")
        if depth < 0:
            raise ValueError("RandAugment depth value must be ≥ 0")
        if augmentation_set not in augmentation_sets.keys():
            raise KeyError(f"RandAugment augmentation_set is not one of {augmentation_sets.keys()}")
        self.severity = severity
        self.depth = depth
        self.augmentation_set = augmentation_sets[augmentation_set]

    def forward(self, img: ImageType) -> ImageType:

        return randaugment(img=img, severity=self.severity, depth=self.depth, augmentation_set=self.augmentation_set)


class RandAugment(Algorithm):
    """
    Object that does RandAugment (Cubuk et al. (2019), RandAugment: Practical
        automated data augmentation with a reduced search space). Can be passed as a
        transform to torchvision.transforms.Compose().

    Args:
        severity (int): Severity of augmentation operators (between 1 to 10). M in the
            original paper. Default = 9.
        depth (int): Depth of augmentation chain. N in the original paper Default = 2.
        augmentation_set (str): One of ["augmentations_all",
            "augmentations_corruption_safe", "augmentations_original"]. Set of
            augmentations to use. "augmentations_corruption_safe" excludes transforms
            that are part of the ImageNet-C/CIFAR10-C test sets.
            "augmentations_original" uses all augmentations, but some of the
            implementations are identical to the original github repo, which appears
            to contain implementation specificities for the augmentations "color",
            "contrast", "sharpness", and "brightness".
    """

    def __init__(self, severity: int = 9, depth: int = 2, augmentation_set: str = "all"):
        if severity < 0 or severity > 10:
            raise ValueError("RandAugment severity value must be 0 ≤ severity ≤ 10")
        if augmentation_set not in augmentation_sets.keys():
            raise KeyError(f"randaugment_augmentation_set is not one of {augmentation_sets.keys()}")
        self.hparams = RandAugmentHparams(severity=severity, depth=depth, augmentation_set=augmentation_set)

    def match(self, event: Event, state: State) -> bool:
        """ Runs on Event.TRAINING_START
        """
        return event == Event.TRAINING_START

    def apply(self, event: Event, state: State, logger: Logger) -> None:
        """ Inserts RandAugment into the list of dataloader transforms
        """
        ra = RandAugmentTransform(**self.hparams.to_dict())
        assert state.train_dataloader is not None
        dataset = state.train_dataloader.dataset
        add_dataset_transform(dataset, ra)
