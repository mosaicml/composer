# Copyright 2021 MosaicML. All Rights Reserved.

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Optional

import torch
import yahp as hp

import composer.algorithms.factorize.factorized_conv as fconv
from composer.algorithms import AlgorithmHparams
from composer.algorithms.factorize.factorize_core import FractionOrInt
from composer.core import Algorithm, Event, Logger, State, surgery

log = logging.getLogger(__name__)

_DEFAULT_MIN_CHANNELS = 256
_DEFAULT_LATENT_CHANNELS = 128
FACTORIZE_LOG_NUM_REPLACEMENTS_KEY = 'factorize/num_modules'


def _log_surgery_result(model: torch.nn.Module):
    new_class = fconv.FactorizedConv2d
    num_replaced_modules = surgery.count_module_instances(model, new_class)
    log.info(f'Applied factorization to model {model.__class__.__name__}. '
             f'Model now has {num_replaced_modules} {new_class.__name__} modules')


def factorize_conv2d_modules(model: torch.nn.Module, min_channels: int, latent_channels: FractionOrInt):

    def _maybe_replace_conv2d(module: torch.nn.Conv2d,
                              module_index: int,
                              min_channels: int = min_channels,
                              latent_channels: FractionOrInt = latent_channels) -> Optional[torch.nn.Module]:
        if min(module.in_channels, module.out_channels) < min_channels:
            return None
        max_rank = fconv.max_rank_with_possible_speedup(module.in_channels, module.out_channels, module.kernel_size)
        latent_channels = fconv.clean_latent_channels(latent_channels, module.in_channels, module.out_channels)
        if max_rank < latent_channels:
            return None  # not enough rank reduction to be worth it
        return fconv.FactorizedConv2d.from_conv2d(module, module_index, latent_channels=latent_channels)

    transforms = {torch.nn.Conv2d: _maybe_replace_conv2d}
    ret = surgery.replace_module_classes(model, policies=transforms)
    _log_surgery_result(model)
    return ret


@dataclass
class FactorizeHparams(AlgorithmHparams):
    min_channels: int = hp.optional(
        doc='Minimum number of channels in a Conv2d layer'
        ' for it to be factorized.',
        default=256,
    )
    latent_channels: float = hp.optional(
        doc='Channel count of latent representations',
        default=128,
    )

    def initialize_object(self) -> Factorize:
        return Factorize(**asdict(self))


class Factorize(Algorithm):

    def __init__(self, min_channels: int = _DEFAULT_MIN_CHANNELS, latent_channels: int = _DEFAULT_LATENT_CHANNELS):
        self.hparams = FactorizeHparams(min_channels=min_channels, latent_channels=latent_channels)

    def match(self, event: Event, state: State) -> bool:
        """Run on Event.INIT

        Args:
            event (:class:`Event`): The current event.
            state (:class:`State`): The current state.

        Returns:
            bool: True if this algorithm should run no
        """
        return event == Event.INIT

    def apply(self, event: Event, state: State, logger: Logger) -> Optional[int]:
        """Apply the Squeeze-and-Excitation layer replacement.

        Args:
            event (Event): the current event
            state (State): the current trainer state
            logger (Logger): the training logger
        """
        assert state.model is not None, "Model must be part of state!"
        factorize_conv2d_modules(state.model,
                                 min_channels=self.hparams.min_channels,
                                 latent_channels=self.hparams.latent_channels)
        _log_surgery_result(state.model)
        num_fconv_modules = surgery.count_module_instances(state.model, fconv.FactorizedConv2d)
        logger.metric_fit({
            FACTORIZE_LOG_NUM_REPLACEMENTS_KEY: num_fconv_modules,
        })
