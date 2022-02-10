# Copyright 2021 MosaicML. All Rights Reserved.

from dataclasses import asdict, dataclass

import yahp as hp

from composer.algorithms.algorithm_hparams import AlgorithmHparams
from composer.algorithms.augmix import AugMix
from composer.algorithms.colout import ColOut
from composer.algorithms.cutmix import CutMix
from composer.algorithms.cutout import CutOut
from composer.algorithms.label_smoothing import LabelSmoothing
from composer.algorithms.mixup import MixUp


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

    def initialize_object(self) -> AugMix:
        return AugMix(**asdict(self))


@dataclass
class ColOutHparams(AlgorithmHparams):
    """See :class:`ColOut`"""
    p_row: float = hp.optional(doc="Fraction of rows to drop", default=0.15)
    p_col: float = hp.optional(doc="Fraction of cols to drop", default=0.15)
    batch: bool = hp.optional(doc="Run ColOut at the batch level", default=True)

    def initialize_object(self) -> ColOut:
        return ColOut(**asdict(self))


@dataclass
class CutMixHparams(AlgorithmHparams):
    """See :class:`CutMix`"""

    num_classes: int = hp.required('Number of classes in the task labels.')
    alpha: float = hp.optional('Strength of interpolation, should be >= 0. No interpolation if alpha=0.', default=1.0)

    def initialize_object(self) -> CutMix:
        return CutMix(**asdict(self))


@dataclass
class CutOutHparams(AlgorithmHparams):
    """See :class:`CutOut`"""

    n_holes: int = hp.optional('Number of holes to cut out', default=1)
    length: int = hp.optional('Side length of the square hole to cut out', default=112)

    def initialize_object(self) -> CutOut:
        return CutOut(**asdict(self))


@dataclass
class LabelSmoothingHparams(AlgorithmHparams):
    """See :class:`LabelSmoothing`"""

    alpha: float = hp.optional(doc='smoothing factor', default=0.1)

    def initialize_object(self) -> LabelSmoothing:
        return LabelSmoothing(**asdict(self))


@dataclass
class MixUpHparams(AlgorithmHparams):
    """See :class:`MixUp`"""

    num_classes: int = hp.required('Number of classes in the task labels.')
    alpha: float = hp.optional('Strength of interpolation, should be >= 0. No interpolation if alpha=0.', default=0.2)

    def initialize_object(self) -> MixUp:
        return MixUp(**asdict(self))
