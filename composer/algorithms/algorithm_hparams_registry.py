# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, Type, Union

import yahp as hp

from composer.algorithms.alibi import Alibi
from composer.algorithms.augmix import AugMix
from composer.algorithms.blurpool import BlurPool
from composer.algorithms.channels_last import ChannelsLast
from composer.algorithms.colout import ColOut
from composer.algorithms.cutmix import CutMix
from composer.algorithms.cutout import CutOut
from composer.algorithms.ema import EMA
from composer.algorithms.factorize import Factorize
from composer.algorithms.fused_layernorm import FusedLayerNorm
from composer.algorithms.gated_linear_units import GatedLinearUnits
from composer.algorithms.ghost_batchnorm import GhostBatchNorm
from composer.algorithms.gradient_clipping import GradientClipping
from composer.algorithms.label_smoothing import LabelSmoothing
from composer.algorithms.layer_freezing import LayerFreezing
from composer.algorithms.mixup import MixUp
from composer.algorithms.no_op_model import NoOpModel
from composer.algorithms.progressive_resizing import ProgressiveResizing
from composer.algorithms.randaugment import RandAugment
from composer.algorithms.sam import SAM
from composer.algorithms.selective_backprop import SelectiveBackprop
from composer.algorithms.seq_length_warmup import SeqLengthWarmup
from composer.algorithms.squeeze_excite import SqueezeExcite
from composer.algorithms.stochastic_depth import StochasticDepth
from composer.algorithms.swa import SWA
from composer.core.algorithm import Algorithm

algorithm_registry: Dict[str, Union[Type[Algorithm], Type[hp.Hparams]]] = {
    'blurpool': BlurPool,
    'channels_last': ChannelsLast,
    'seq_length_warmup': SeqLengthWarmup,
    'cutmix': CutMix,
    'cutout': CutOut,
    'ema': EMA,
    'factorize': Factorize,
    'fused_layernorm': FusedLayerNorm,
    'gated_linear_units': GatedLinearUnits,
    'ghost_batchnorm': GhostBatchNorm,
    'label_smoothing': LabelSmoothing,
    'layer_freezing': LayerFreezing,
    'squeeze_excite': SqueezeExcite,
    'swa': SWA,
    'no_op_model': NoOpModel,
    'mixup': MixUp,
    'stochastic_depth': StochasticDepth,
    'colout': ColOut,
    'progressive_resizing': ProgressiveResizing,
    'randaugment': RandAugment,
    'augmix': AugMix,
    'sam': SAM,
    'alibi': Alibi,
    'selective_backprop': SelectiveBackprop,
    'gradient_clipping': GradientClipping,
}
