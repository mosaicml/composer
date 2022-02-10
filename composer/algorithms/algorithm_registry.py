# Copyright 2021 MosaicML. All Rights Reserved.

from typing import Dict, List, Type

from composer.algorithms.algorithm_hparams import AlgorithmHparams
from composer.algorithms.alibi import AlibiHparams
from composer.algorithms.blurpool import BlurPoolHparams
from composer.algorithms.channels_last import ChannelsLastHparams
# from composer.algorithms.cutmix import CutMixHparams
# from composer.algorithms.cutout import CutOutHparams
from composer.algorithms.factorize import FactorizeHparams
from composer.algorithms.ghost_batchnorm import GhostBatchNormHparams
# from composer.algorithms.hparams import AlibiHparams
# from composer.algorithms.hparams import BlurPoolHparams
# from composer.algorithms.hparams import ChannelsLastHparams
from composer.algorithms.hparams import (AugMixHparams, ColOutHparams, CutMixHparams, CutOutHparams,
                                         LabelSmoothingHparams, MixUpHparams, RandAugmentHparams)
from composer.algorithms.layer_freezing import LayerFreezingHparams
from composer.algorithms.no_op_model import NoOpModelHparams
from composer.algorithms.progressive_resizing import ProgressiveResizingHparams
from composer.algorithms.sam import SAMHparams
from composer.algorithms.scale_schedule import ScaleScheduleHparams
from composer.algorithms.selective_backprop import SelectiveBackpropHparams
from composer.algorithms.seq_length_warmup import SeqLengthWarmupHparams
from composer.algorithms.squeeze_excite import SqueezeExciteHparams
from composer.algorithms.stochastic_depth import StochasticDepthHparams
from composer.algorithms.swa.hparams import SWAHparams
from composer.core.algorithm import Algorithm

# from composer.algorithms.hparams import FactorizeHparams
# from composer.algorithms.hparams import GhostBatchNormHparams
# from composer.algorithms.hparams import LabelSmoothingHparams
# from composer.algorithms.hparams import LayerFreezingHparams
# from composer.algorithms.hparams import MixUpHparams
# from composer.algorithms.hparams import NoOpModelHparams
# from composer.algorithms.hparams import ProgressiveResizingHparams
# from composer.algorithms.hparams import RandAugmentHparams
# from composer.algorithms.hparams import SAMHparams
# from composer.algorithms.hparams import ScaleScheduleHparams
# from composer.algorithms.hparams import SelectiveBackpropHparams
# from composer.algorithms.hparams import SeqLengthWarmupHparams
# from composer.algorithms.hparams import SqueezeExciteHparams
# from composer.algorithms.hparams import StochasticDepthHparams
# from composer.algorithms.hparams import SWAHparams

registry: Dict[str, Type[AlgorithmHparams]] = {
    'blurpool': BlurPoolHparams,
    'channels_last': ChannelsLastHparams,
    'seq_length_warmup': SeqLengthWarmupHparams,
    'cutmix': CutMixHparams,
    'cutout': CutOutHparams,
    'factorize': FactorizeHparams,
    'ghost_batchnorm': GhostBatchNormHparams,
    'label_smoothing': LabelSmoothingHparams,
    'layer_freezing': LayerFreezingHparams,
    'squeeze_excite': SqueezeExciteHparams,
    'swa': SWAHparams,
    'no_op_model': NoOpModelHparams,
    'mixup': MixUpHparams,
    'scale_schedule': ScaleScheduleHparams,
    'stochastic_depth': StochasticDepthHparams,
    'colout': ColOutHparams,
    'progressive_resizing': ProgressiveResizingHparams,
    'randaugment': RandAugmentHparams,
    'augmix': AugMixHparams,
    'sam': SAMHparams,
    'alibi': AlibiHparams,
    'selective_backprop': SelectiveBackpropHparams,
}


def get_algorithm_registry():
    return registry


def get_algorithm(params: AlgorithmHparams) -> Algorithm:
    return params.initialize_object()


def list_algorithms() -> List[str]:
    return list(registry.keys())
