# Copyright 2021 MosaicML. All Rights Reserved.

import textwrap
from dataclasses import asdict, dataclass
from typing import Optional

import yahp as hp

from composer.algorithms.algorithm_hparams import AlgorithmHparams
from composer.algorithms.alibi import Alibi
from composer.algorithms.augmix import AugMix
from composer.algorithms.blurpool import BlurPool
from composer.algorithms.channels_last import ChannelsLast
from composer.algorithms.colout import ColOut
from composer.algorithms.cutmix import CutMix
from composer.algorithms.cutout import CutOut
from composer.algorithms.factorize import Factorize
from composer.algorithms.ghost_batchnorm import GhostBatchNorm
from composer.algorithms.label_smoothing import LabelSmoothing
from composer.algorithms.layer_freezing import LayerFreezing
from composer.algorithms.mixup import MixUp
from composer.algorithms.no_op_model import NoOpModel
from composer.algorithms.progressive_resizing import ProgressiveResizing
from composer.algorithms.randaugment import RandAugment
from composer.algorithms.sam import SAM
from composer.algorithms.scale_schedule import ScaleSchedule
from composer.algorithms.selective_backprop import SelectiveBackprop
from composer.algorithms.seq_length_warmup import SeqLengthWarmup
from composer.algorithms.squeeze_excite import SqueezeExcite
from composer.algorithms.stochastic_depth import StochasticDepth
from composer.algorithms.stochastic_depth.stochastic_depth import (_STOCHASTIC_LAYER_MAPPING,
                                                                   _validate_stochastic_hparams)
from composer.algorithms.swa import SWA


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
class ChannelsLastHparams(AlgorithmHparams):
    """ChannelsLast has no hyperparameters, so this class has no member variables."""

    def initialize_object(self) -> ChannelsLast:
        return ChannelsLast()


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
    uniform_sampling: bool = hp.optional('Mix pixels with uniform probability', default=False)

    def initialize_object(self) -> CutMix:
        return CutMix(**asdict(self))


@dataclass
class CutOutHparams(AlgorithmHparams):
    """See :class:`CutOut`"""

    num_holes: int = hp.optional('Number of holes to cut out', default=1)
    length: float = hp.optional('Relative or absolute side length of the square hole to cut out', default=0.5)
    uniform_sampling: bool = hp.optional('Mask pixels with uniform probability', default=False)

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

    smoothing: float = hp.optional(doc='smoothing factor between 0 and 1', default=0.1)

    def initialize_object(self) -> LabelSmoothing:
        return LabelSmoothing(**asdict(self))


@dataclass
class LayerFreezingHparams(AlgorithmHparams):
    """See :class:`LayerFreezing`"""

    freeze_start: float = hp.optional(doc='The percentage of epochs to run before freezing begins.', default=0.5)
    freeze_level: float = hp.optional(doc='Scale factor for the percentage of the network to freeze.', default=1.0)

    def initialize_object(self) -> LayerFreezing:
        return LayerFreezing(**asdict(self))


@dataclass
class MixUpHparams(AlgorithmHparams):
    """See :class:`MixUp`"""

    num_classes: int = hp.required('Number of classes in the task labels.')
    alpha: float = hp.optional('Strength of interpolation, should be >= 0. No interpolation if alpha=0.', default=0.2)

    def initialize_object(self) -> MixUp:
        return MixUp(**asdict(self))


@dataclass
class NoOpModelHparams(AlgorithmHparams):

    def initialize_object(self) -> NoOpModel:
        return NoOpModel()


@dataclass
class ProgressiveResizingHparams(AlgorithmHparams):
    """See :class:`ProgressiveResizing`"""

    mode: str = hp.optional(doc="Type of scaling to perform", default="resize")
    initial_scale: float = hp.optional(doc="Initial scale factor", default=0.5)
    finetune_fraction: float = hp.optional(doc="Fraction of training to reserve for finetuning on full-sized inputs",
                                           default=0.2)
    resize_targets: bool = hp.optional(doc="Also resize targets", default=False)

    def initialize_object(self) -> ProgressiveResizing:
        return ProgressiveResizing(**asdict(self))


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


@dataclass
class SAMHparams(AlgorithmHparams):
    """See :class:`SAM`"""
    rho: float = hp.optional(doc='The neighborhood size parameter of SAM. Must be greater than 0.', default=0.05)
    epsilon: float = hp.optional(doc='A small value added to gradient norm for numerical stability.', default=1.0e-12)
    interval: int = hp.optional(doc='SAM will run once per `interval` steps. A value of 1 will cause'
                                'SAM to run every step. Steps on which SAM runs take roughly twice'
                                'as much time to complete.',
                                default=1)

    def initialize_object(self) -> SAM:
        return SAM(**asdict(self))


@dataclass
class ScaleScheduleHparams(AlgorithmHparams):
    """See :class:`ScaleSchedule`"""

    ratio: float = hp.optional('Ratio to scale the schedule.', default=1.0)

    def initialize_object(self) -> "ScaleSchedule":
        return ScaleSchedule(**asdict(self))


@dataclass
class SelectiveBackpropHparams(AlgorithmHparams):
    """See :class:`SelectiveBackprop`"""

    start: float = hp.optional(doc="SB interval start, as fraction of training duration", default=0.5)
    end: float = hp.optional(doc="SB interval end, as fraction of training duration", default=0.9)
    keep: float = hp.optional(doc="fraction of minibatch to select and keep for gradient computation", default=0.5)
    scale_factor: float = hp.optional(doc="scale for downsampling input for selection forward pass", default=0.5)
    interrupt: int = hp.optional(doc="interrupt SB with a vanilla minibatch step every 'interrupt' batches", default=2)

    def initialize_object(self) -> SelectiveBackprop:
        return SelectiveBackprop(**asdict(self))


@dataclass
class SeqLengthWarmupHparams(AlgorithmHparams):

    duration: float = hp.optional("Fraction of total training time to apply sequential length warmup learning.",
                                  default=0.3)
    min_seq_length: int = hp.optional("Starting sequence length.", default=8)
    max_seq_length: int = hp.optional("End sequence length", default=1024)
    step_size: int = hp.optional("Sequence length step size", default=8)
    truncate: bool = hp.optional("Truncate tensors or reshape extra tokens to new examples.", default=True)

    def initialize_object(self) -> "SeqLengthWarmup":
        return SeqLengthWarmup(**asdict(self))


@dataclass
class StochasticDepthHparams(AlgorithmHparams):
    """See :class:`StochasticDepth`"""

    target_layer_name: str = hp.required(
        f'Reference name of layer to replace. "block" method can be {list(_STOCHASTIC_LAYER_MAPPING["block"].keys())}.'
        f' "sample" method can be {list(_STOCHASTIC_LAYER_MAPPING["sample"].keys())}.')
    stochastic_method: str = hp.optional('The version of stochastic depth to use. One of ["sample", "block"].',
                                         default='block')
    drop_rate: float = hp.optional('The probability of dropping a block or sample.', default=0.2)
    drop_distribution: str = hp.optional(
        '"Uniform" keeps the drop rate the same across blocks. "linear" linearly'
        ' increases the drop rate with block depth until it reaches `drop_rate`.',
        default='linear')
    use_same_gpu_seed: bool = hp.optional(
        'Whether or not to drop the same blocks across GPUs. Only used with "block" method.', default=True)
    drop_warmup: str = hp.optional(textwrap.dedent("""\
            Time string to represent the amount of training to warmup the `drop_rate`.
            Only use with "block" stochastic method."""),
                                   default="0dur")

    def initialize_object(self) -> StochasticDepth:
        return StochasticDepth(**asdict(self))

    def validate(self):
        super().validate()
        _validate_stochastic_hparams(target_layer_name=self.target_layer_name,
                                     stochastic_method=self.stochastic_method,
                                     drop_rate=self.drop_rate,
                                     drop_distribution=self.drop_distribution,
                                     drop_warmup=self.drop_warmup)


@dataclass
class SqueezeExciteHparams(AlgorithmHparams):
    """See :class:`SqueezeExcite`"""

    latent_channels: float = hp.optional(
        doc='Dimensionality of hidden layer within the added MLP.',
        default=64,
    )
    min_channels: int = hp.optional(
        doc='Minimum number of channels in a Conv2d layer'
        ' for a squeeze-excite block to be placed after it.',
        default=128,
    )

    def initialize_object(self) -> SqueezeExcite:
        return SqueezeExcite(**asdict(self))


@dataclass
class SWAHparams(AlgorithmHparams):
    """See :class:`~.composer.algorithms.swa.SWA`"""

    swa_start: str = hp.optional(
        doc='Time string denoting the amount of training '
        'completed before stochastic weight averaging begins. Currently only units of '
        'duration (e.g. "0.7dur") and epoch (e.g "50ep") are supported.',
        default="0.7dur",
    )
    swa_end: str = hp.optional(
        doc='Time string denoting amount of training completed before the baseline '
        '(non-averaged) model is replaced with the stochastic weight averaged model. '
        'Currently only units of duration (e.g. "0.97dur") and epoch (e.g "88ep") are supported.',
        default="0.97dur")
    update_interval: str = hp.optional(doc='Time string denoting how often the averaged model is updated. For example, '
                                       '"1ep" means the averaged model will be updated once per epoch, and '
                                       '"10ba" means the averaged model will be updated every 10 batches.',
                                       default="1ep")
    schedule_swa_lr: bool = hp.optional(doc='Flag to determine whether to apply an SWA-specific LR schedule during the '
                                        'period in which SWA is active.',
                                        default=False)
    anneal_strategy: str = hp.optional(doc='SWA learning rate annealing schedule strategy. '
                                       '"linear" for linear annealing, "cos" for cosine annealing.',
                                       default='linear')
    anneal_steps: int = hp.optional(
        doc='Number of SWA model updates over which to anneal SWA learning rate. Note '
        'that updates are determined by the ``update_interval`` argument.',
        default=10,
    )
    swa_lr: Optional[float] = hp.optional(
        doc='The final learning rate to anneal towards with this scheduler. '
        'Set to None for no annealing.',
        default=None,
    )

    def initialize_object(self):
        return SWA(**asdict(self))
