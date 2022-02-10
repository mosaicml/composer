# Copyright 2021 MosaicML. All Rights Reserved.

from dataclasses import asdict, dataclass
from typing import Optional

import yahp as hp

from composer.algorithms.algorithm_hparams import AlgorithmHparams
from composer.algorithms.alibi import Alibi
from composer.algorithms.augmix import AugMix
from composer.algorithms.blurpool import BlurPool
from composer.algorithms.colout import ColOut
from composer.algorithms.cutmix import CutMix
from composer.algorithms.cutout import CutOut
from composer.algorithms.factorize import Factorize
from composer.algorithms.ghost_batchnorm import GhostBatchNorm
from composer.algorithms.label_smoothing import LabelSmoothing
from composer.algorithms.mixup import MixUp
from composer.algorithms.randaugment import RandAugment


@dataclass
class AlibiHparams(AlgorithmHparams):
    """See :class:`Alibi`"""

    position_embedding_attribute: str = hp.required("attribute name of position embeddings within the model. "
                                                    "For example in HuggingFace's GPT2, the position "
                                                    "embeddings are 'transformer.wpe'")
    attention_module_name: str = hp.required("module/class that will have its self-attention "
                                             "function replaced. For example, in HuggingFace's "
                                             "GPT, the self-attention module is "
                                             "'transformers.models.gpt2.modeling_gpt2.GPT2Attention'")
    attr_to_replace: str = hp.required("model attribute that self-attention function will "
                                       "replace. For example, in HuggingFace's "
                                       "GPT2, the self-attention function is '_attn'")
    alibi_attention: str = hp.required("new self-attention function in which ALiBi is "
                                       "implemented. Used to replace "
                                       "'{attention_module}.{attr_to_replace}'")
    mask_replacement_function: Optional[str] = hp.optional(
        "function to replace model's attention mask. This is "
        "sometimes necessary for evaluating on sequence "
        " lengths longer than the model was initialized to accommodate.",
        default=None)
    heads_per_layer: Optional[int] = hp.optional(
        'Number of attention heads per layer. If '
        '"None", will attempt to determine from model.config.n_head.',
        default=None)
    max_sequence_length: int = hp.optional('Maximum allowable sequence length', default=8192)
    train_sequence_length_scaling: float = hp.optional(
        'Amount by which to scale training sequence length. One batch of training data '
        'will be reshaped from size (sequence_length, batch) to '
        '(sequence_length*train_sequence_length_scaling, batch/train_sequence_length_scaling)',
        default=0.25)

    def initialize_object(self) -> "Alibi":
        return Alibi(**asdict(self))


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
class BlurPoolHparams(AlgorithmHparams):
    """See :class:`BlurPool`"""

    replace_convs: bool = hp.optional('Replace Conv2d with BlurConv2d if stride > 1', default=True)
    replace_maxpools: bool = hp.optional('Replace MaxPool2d with BlurMaxPool2d', default=True)
    blur_first: bool = hp.optional('Blur input before convolution', default=True)

    def initialize_object(self) -> "BlurPool":
        return BlurPool(**asdict(self))


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
class FactorizeHparams(AlgorithmHparams):
    """See :class:`Factorize`"""
    factorize_convs: bool = hp.optional(
        doc='Whether to factorize convolutional layers',
        default=True,
    )
    factorize_linears: bool = hp.optional(
        doc='Whether to factorize linear layers',
        default=True,
    )
    min_channels: int = hp.optional(
        doc=('Minimum number of channels in a Conv2d module' + ' for it to be factorized.'),
        default=512,
    )
    latent_channels: float = hp.optional(
        doc='Number or relative fraction of channels in factorized convolution latent representations',
        default=0.25,
    )
    min_features: int = hp.optional(
        doc=('Minimum number of features in a Linear module' + ' for it to be factorized.'),
        default=512,
    )
    latent_features: float = hp.optional(
        doc='Number or relative fraction of features in factorized linear latent representations',
        default=0.25,
    )

    def initialize_object(self) -> Factorize:
        return Factorize(**asdict(self))


@dataclass
class GhostBatchNormHparams(AlgorithmHparams):
    """See :class:`GhostBatchNorm`"""

    ghost_batch_size: int = hp.optional(doc='Size of sub-batches to normalize over', default=32)

    def initialize_object(self) -> GhostBatchNorm:
        return GhostBatchNorm(**asdict(self))


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


@dataclass
class RandAugmentHparams(AlgorithmHparams):
    """See :class:`RandAugment`"""

    severity: int = hp.optional(doc="Intensity of each augmentation. Ranges from 0 (none) to 10 (maximum)", default=9)
    depth: int = hp.optional(doc="Number of augmentations to compose in a row", default=2)
    augmentation_set: str = hp.optional(
        doc=
        "Set of augmentations to sample from. 'all', 'safe' (only augmentations that don't appear on CIFAR10C/ImageNet10C), or 'original'",
        default="all")

    def initialize_object(self) -> "RandAugment":
        return RandAugment(**asdict(self))
