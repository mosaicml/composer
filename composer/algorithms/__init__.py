# Copyright 2021 MosaicML. All Rights Reserved.

from composer.algorithms.algorithm_hparams import AlgorithmHparams as AlgorithmHparams
from composer.algorithms.algorithm_registry import get_algorithm_registry as get_algorithm_registry
from composer.algorithms.algorithm_registry import list_algorithms as list_algorithms
from composer.algorithms.alibi import AlibiHparams as AlibiHparams
from composer.algorithms.augmix import AugMix as AugMix
from composer.algorithms.augmix import AugMixHparams as AugMixHparams
from composer.algorithms.blurpool import BlurPool as BlurPool
from composer.algorithms.blurpool import BlurPoolHparams as BlurPoolHparams
from composer.algorithms.channels_last import ChannelsLast as ChannelsLast
from composer.algorithms.channels_last import ChannelsLastHparams as ChannelsLastHparams
from composer.algorithms.colout import ColOut as ColOut
from composer.algorithms.colout import ColOutHparams as ColOutHparams
from composer.algorithms.cutout import CutOut as CutOut
from composer.algorithms.cutout import CutOutHparams as CutOutHparams
from composer.algorithms.dummy import Dummy as Dummy
from composer.algorithms.dummy import DummyHparams as DummyHparams
from composer.algorithms.ghost_batchnorm import GhostBatchNorm as GhostBatchNorm
from composer.algorithms.ghost_batchnorm import GhostBatchNormHparams as GhostBatchNormHparams
from composer.algorithms.label_smoothing import LabelSmoothing as LabelSmoothing
from composer.algorithms.label_smoothing import LabelSmoothingHparams as LabelSmoothingHparams
from composer.algorithms.layer_freezing import LayerFreezing as LayerFreezing
from composer.algorithms.layer_freezing import LayerFreezingHparams as LayerFreezingHparams
from composer.algorithms.mixup import MixUp as MixUp
from composer.algorithms.mixup import MixUpHparams as MixUpHparams
from composer.algorithms.no_op_model import NoOpModel as NoOpModel
from composer.algorithms.no_op_model import NoOpModelHparams as NoOpModelHparams
from composer.algorithms.progressive_resizing import ProgressiveResizing as ProgressiveResizing
from composer.algorithms.progressive_resizing import ProgressiveResizingHparams as ProgressiveResizingHparams
from composer.algorithms.randaugment import RandAugment as RandAugment
from composer.algorithms.randaugment import RandAugmentHparams as RandAugmentHparams
from composer.algorithms.sam import SAM as SAM
from composer.algorithms.sam import SAMHparams as SAMHparams
from composer.algorithms.scale_schedule import ScaleSchedule as ScaleSchedule
from composer.algorithms.scale_schedule import ScaleScheduleHparams as ScaleScheduleHparams
from composer.algorithms.selective_backprop import SelectiveBackprop as SelectiveBackprop
from composer.algorithms.selective_backprop import SelectiveBackpropHparams as SelectiveBackpropHparams
from composer.algorithms.seq_length_warmup import SeqLengthWarmup as SeqLengthWarmup
from composer.algorithms.seq_length_warmup import SeqLengthWarmupHparams as SeqLengthWarmupHparams
from composer.algorithms.squeeze_excite import SqueezeExcite as SqueezeExcite
from composer.algorithms.squeeze_excite import SqueezeExcite2d as SqueezeExcite2d
from composer.algorithms.squeeze_excite import SqueezeExciteConv2d as SqueezeExciteConv2d
from composer.algorithms.squeeze_excite import SqueezeExciteHparams as SqueezeExciteHparams
from composer.algorithms.stochastic_depth import StochasticDepth as StochasticDepth
from composer.algorithms.stochastic_depth import StochasticDepthHparams as StochasticDepthHparams
from composer.algorithms.swa import SWA as SWA
from composer.algorithms.swa.hparams import SWAHparams as SWAHparams

load_multiple = AlgorithmHparams.load_multiple
load = AlgorithmHparams.load
