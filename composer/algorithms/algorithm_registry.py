# Copyright 2021 MosaicML. All Rights Reserved.

from typing import Dict, List, Type

from composer.algorithms.algorithm_hparams import AlgorithmHparams
from composer.algorithms.hparams import (AGCHparams, AlibiHparams, AugMixHparams, BlurPoolHparams, ChannelsLastHparams,
                                         ColOutHparams, CutMixHparams, CutOutHparams, EMAHparams, FactorizeHparams,
                                         GhostBatchNormHparams, LabelSmoothingHparams, LayerFreezingHparams,
                                         MixUpHparams, NoOpModelHparams, ProgressiveResizingHparams, RandAugmentHparams,
                                         SAMHparams, ScaleScheduleHparams, SelectiveBackpropHparams,
                                         SeqLengthWarmupHparams, SqueezeExciteHparams, StochasticDepthHparams,
                                         SWAHparams)
from composer.core.algorithm import Algorithm

registry: Dict[str, Type[AlgorithmHparams]] = {
    'blurpool': BlurPoolHparams,
    'channels_last': ChannelsLastHparams,
    'seq_length_warmup': SeqLengthWarmupHparams,
    'cutmix': CutMixHparams,
    'cutout': CutOutHparams,
    'ema': EMAHparams,
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
    'agc': AGCHparams,
}


def get_algorithm_registry():
    return registry


def get_algorithm(params: AlgorithmHparams) -> Algorithm:
    return params.initialize_object()


def list_algorithms() -> List[str]:
    return list(registry.keys())
