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
from composer.algorithms.agc import AGC
from composer.algorithms.algorithm_hparams import AlgorithmHparams
from composer.algorithms.algorithm_registry import get_algorithm_registry, list_algorithms
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
from composer.algorithms.hparams import (AGCHparams, AlibiHparams, AugMixHparams, BlurPoolHparams, ChannelsLastHparams,
                                         ColOutHparams, CutMixHparams, CutOutHparams, EMAHparams, FactorizeHparams,
                                         GhostBatchNormHparams, LabelSmoothingHparams, LayerFreezingHparams,
                                         MixUpHparams, NoOpModelHparams, ProgressiveResizingHparams, RandAugmentHparams,
                                         SAMHparams, SelectiveBackpropHparams, SeqLengthWarmupHparams,
                                         SqueezeExciteHparams, StochasticDepthHparams, SWAHparams)
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

load_multiple = AlgorithmHparams.load_multiple
load = AlgorithmHparams.load

__all__ = [
    "load",
    "load_multiple",
    "get_algorithm_registry",
    "list_algorithms",
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

    # hparams objects
    "AGCHparams",
    "AlgorithmHparams",
    "AlibiHparams",
    "AugMixHparams",
    "BlurPoolHparams",
    "ChannelsLastHparams",
    "ColOutHparams",
    "CutMixHparams",
    "CutOutHparams",
    "EMAHparams",
    "FactorizeHparams",
    "GhostBatchNormHparams",
    "LabelSmoothingHparams",
    "LayerFreezingHparams",
    "MixUpHparams",
    "NoOpModelHparams",
    "ProgressiveResizingHparams",
    "RandAugmentHparams",
    "SAMHparams",
    "SelectiveBackpropHparams",
    "SeqLengthWarmupHparams",
    "SqueezeExciteHparams",
    "StochasticDepthHparams",
    "SWAHparams",
]
