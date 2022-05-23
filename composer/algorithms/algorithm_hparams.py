# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import os
from typing import Dict, List, Optional, Type, Union

import yahp as hp

import composer
from composer.algorithms.agc import AGC
from composer.algorithms.alibi import Alibi
from composer.algorithms.augmix import AugMix
from composer.algorithms.blurpool import BlurPool
from composer.algorithms.channels_last import ChannelsLast
from composer.algorithms.colout import ColOut
from composer.algorithms.cutmix import CutMix
from composer.algorithms.cutout import CutOut
from composer.algorithms.ema import EMA
from composer.algorithms.factorize import Factorize
from composer.algorithms.ghost_batchnorm import GhostBatchNorm
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
    'agc': AGC,
}


def load(algorithm_cls: Union[Type[Algorithm], Type[hp.Hparams]], alg_params: Optional[str]) -> Algorithm:
    inverted_registry = {v: k for (k, v) in algorithm_registry.items()}
    alg_name = inverted_registry[algorithm_cls]
    alg_folder = os.path.join(os.path.dirname(composer.__file__), "yamls", "algorithms")
    if alg_params is None:
        hparams_file = os.path.join(alg_folder, f"{alg_name}.yaml")
    else:
        hparams_file = os.path.join(alg_folder, alg_name, f"{alg_params}.yaml")
    alg = hp.create(algorithm_cls, f=hparams_file, cli_args=False)
    assert isinstance(alg, Algorithm)
    return alg


def load_multiple(cls, *algorithms: str) -> List[Algorithm]:
    algs = []
    for alg in algorithms:
        alg_parts = alg.split("/")
        alg_name = alg_parts[0]
        if len(alg_parts) > 1:
            alg_params = "/".join(alg_parts[1:])
        else:
            alg_params = None
        try:
            alg = algorithm_registry[alg_name]
        except KeyError as e:
            raise ValueError(f"Algorithm {e.args[0]} not found") from e
        algs.append(load(alg, alg_params))
    return algs
