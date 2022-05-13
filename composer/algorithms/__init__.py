# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Efficiency methods for training.

Examples include :class:`~composer.algorithms.label_smoothing.LabelSmoothing`
and adding :class:`~composer.algorithms.squeeze_excite.SqueezeExcite` blocks,
among many others.

Algorithms are implemented in both a standalone functional form (see :mod:`composer.functional`)
and as subclasses of :class:`Algorithm` for integration in the Composer :class:`Trainer`.
The former are easier to integrate piecemeal into an existing codebase.
The latter are easier to compose together, since they all have the same public interface
and work automatically with the Composer :py:class:`~composer.trainer.Trainer`.

For ease of composability, algorithms in our Trainer are based on the two-way callbacks concept from
`Howard et al, 2020 <https://arxiv.org/abs/2002.04688>`_. Each algorithm implements two methods:

* :meth:`Algorithm.match`: returns ``True`` if the algorithm should be run given the current
  :class:`State` and :class:`~composer.core.event.Event`.
* :meth:`Algorithm.apply`: performs an in-place modification of the given
  :class:`State`

For example, a simple algorithm that shortens training:

.. code-block:: python

    from composer import Algorithm, State, Event, Logger

    class ShortenTraining(Algorithm):

        def match(self, state: State, event: Event, logger: Logger) -> bool:
            return event == Event.INIT

        def apply(self, state: State, event: Event, logger: Logger):
            state.max_duration /= 2  # cut training time in half

For more information about events, see :class:`~composer.core.event.Event`.
"""
import os
from typing import List, Optional, Type

import yahp as hp

import composer
from composer.algorithms.agc import AGC
from composer.algorithms.alibi import Alibi
from composer.algorithms.augmix import AugmentAndMixTransform, AugMix
from composer.algorithms.blurpool import BlurPool
from composer.algorithms.channels_last import ChannelsLast
from composer.algorithms.colout import ColOut, ColOutTransform
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
from composer.algorithms.randaugment import RandAugment, RandAugmentTransform
from composer.algorithms.sam import SAM
from composer.algorithms.selective_backprop import SelectiveBackprop
from composer.algorithms.seq_length_warmup import SeqLengthWarmup
from composer.algorithms.squeeze_excite import SqueezeExcite, SqueezeExcite2d, SqueezeExciteConv2d
from composer.algorithms.stochastic_depth import StochasticDepth
from composer.algorithms.swa import SWA
from composer.core.algorithm import Algorithm

algorithm_registry = {
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


def load(algorithm_cls: Type[Algorithm], alg_params: Optional[str]) -> Algorithm:
    inverted_registry = {v: k for (k, v) in algorithm_registry.items()}
    alg_name = inverted_registry[algorithm_cls]
    alg_folder = os.path.join(os.path.dirname(composer.__file__), "yamls", "algorithms")
    if alg_params is None:
        hparams_file = os.path.join(alg_folder, f"{alg_name}.yaml")
    else:
        hparams_file = os.path.join(alg_folder, alg_name, f"{alg_params}.yaml")
    return hp.create(algorithm_cls, f=hparams_file, cli_args=False)


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


__all__ = [
    "AGC",
    "Alibi",
    "AugmentAndMixTransform",
    "AugMix",
    "BlurPool",
    "ChannelsLast",
    "ColOut",
    "ColOutTransform",
    "CutMix",
    "CutOut",
    "EMA",
    "Factorize",
    "GhostBatchNorm",
    "LabelSmoothing",
    "LayerFreezing",
    "MixUp",
    "NoOpModel",
    "ProgressiveResizing",
    "RandAugment",
    "RandAugmentTransform",
    "SAM",
    "SelectiveBackprop",
    "SeqLengthWarmup",
    "SqueezeExcite",
    "SqueezeExcite2d",
    "SqueezeExciteConv2d",
    "StochasticDepth",
    "SWA",
    "algorithm_registry",
]
