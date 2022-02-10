# Copyright 2021 MosaicML. All Rights Reserved.

"""Algorithms programmatically modify the model or training process.

Examples include :class:`smoothing the labels <composer.algorithms.label_smoothing.LabelSmoothing>`
and adding :class:`Squeeze-and-Excitation <composer.algorithms.squeeze_excite.SqueezeExcite>` blocks,
among many others.

Algorithms are implemented in both a standalone functional form (see :mod:`composer.functional`)
and as subclasses of :class:`Algorithm` for integration in the Composer :class:`Trainer`.
The former are easier to integrate piecemeal into an existing codebase.
The latter are easier to compose together, since they all have the same public interface
and work automatically with the Composer :py:class:`~composer.trainer.Trainer`.

For ease of composability, algorithms in our Trainer are based on the two-way callbacks concept from
`Howard et al., 2020 <https://arxiv.org/abs/2002.04688>`_. Each algorithm implements two methods:

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

For reference, available events include:

.. include:: event_table.rst

For more information about events, see :class:`~composer.core.event.Event`.
"""
from composer.algorithms.algorithm_hparams import AlgorithmHparams as AlgorithmHparams
from composer.algorithms.algorithm_registry import get_algorithm_registry as get_algorithm_registry
from composer.algorithms.algorithm_registry import list_algorithms as list_algorithms
from composer.algorithms.alibi import AlibiHparams as AlibiHparams
from composer.algorithms.augmix import AugMix as AugMix
from composer.algorithms.blurpool import BlurPool as BlurPool
from composer.algorithms.blurpool import BlurPoolHparams as BlurPoolHparams
from composer.algorithms.channels_last import ChannelsLast as ChannelsLast
from composer.algorithms.channels_last import ChannelsLastHparams as ChannelsLastHparams
from composer.algorithms.colout import ColOut as ColOut
from composer.algorithms.cutmix import CutMix as CutMix
from composer.algorithms.cutout import CutOut as CutOut
from composer.algorithms.factorize import Factorize as Factorize
from composer.algorithms.factorize import FactorizeHparams as FactorizeHparams
from composer.algorithms.ghost_batchnorm import GhostBatchNorm as GhostBatchNorm
from composer.algorithms.ghost_batchnorm import GhostBatchNormHparams as GhostBatchNormHparams
from composer.algorithms.hparams import AugMixHparams as AugMixHparams
from composer.algorithms.hparams import ColOutHparams as ColOutHparams
from composer.algorithms.hparams import CutMixHparams as CutMixHparams
from composer.algorithms.hparams import CutOutHparams as CutOutHparams
from composer.algorithms.hparams import LabelSmoothingHparams as LabelSmoothingHparams
from composer.algorithms.hparams import MixUpHparams as MixUpHparams
from composer.algorithms.hparams import RandAugmentHparams as RandAugmentHparams
from composer.algorithms.label_smoothing import LabelSmoothing as LabelSmoothing
from composer.algorithms.layer_freezing import LayerFreezing as LayerFreezing
from composer.algorithms.layer_freezing import LayerFreezingHparams as LayerFreezingHparams
from composer.algorithms.mixup import MixUp as MixUp
from composer.algorithms.no_op_model import NoOpModel as NoOpModel
from composer.algorithms.no_op_model import NoOpModelHparams as NoOpModelHparams
from composer.algorithms.progressive_resizing import ProgressiveResizing as ProgressiveResizing
from composer.algorithms.progressive_resizing import ProgressiveResizingHparams as ProgressiveResizingHparams
from composer.algorithms.randaugment import RandAugment as RandAugment
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
