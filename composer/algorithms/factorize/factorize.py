# Copyright 2021 MosaicML. All Rights Reserved.

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import List, Optional, Tuple, TypeVar

import torch
import yahp as hp

from composer.algorithms import AlgorithmHparams
from composer.algorithms.factorize.modules import (FactorizedConv2d, FactorizedLinear, FractionOrInt,
                                                   factorizing_could_speedup)
from composer.core import Algorithm, Event, Logger, State, surgery

log = logging.getLogger(__name__)

_DEFAULT_SHOULD_FACTORIZE_CONVS = True
_DEFAULT_SHOULD_FACTORIZE_LINEARS = True
_DEFAULT_MIN_CHANNELS = 256
_DEFAULT_MIN_FEATURES = 256
_DEFAULT_LATENT_CHANNELS = 128
_DEFAULT_LATENT_FEATURES = 128
LOG_NUM_CONV2D_REPLACEMENTS_KEY = 'factorize/num_conv2d_replacements'
LOG_NUM_LINEAR_REPLACEMENTS_KEY = 'factorize/num_linear_replacements'


def _python_log_surgery_result(model: torch.nn.Module, new_class: TypeVar):
    num_replaced_modules = surgery.count_module_instances(model, new_class)
    log.info(f'Applied factorization to model {model.__class__.__name__}. '
             f'Model now has {num_replaced_modules} {new_class.__name__} modules')


def _replace_module_class_in_model(
        model: torch.nn.Conv2d, module_class: TypeVar,
        f_replace: surgery.ReplacementFunction) -> List[Tuple[torch.nn.Module, torch.nn.Module]]:
    transforms = {module_class: f_replace}
    ret = surgery.replace_module_classes(model, policies=transforms)
    _python_log_surgery_result(model, module_class)
    return ret


def factorize_conv2d_modules(model: torch.nn.Module, min_channels: int, latent_channels: FractionOrInt):

    def _maybe_replace_conv2d(module: torch.nn.Conv2d,
                              module_index: int,
                              min_channels: int = min_channels,
                              latent_channels: FractionOrInt = latent_channels) -> Optional[torch.nn.Module]:
        wide_enough = min(module.out_channels, module.in_channels) >= min_channels
        if factorizing_could_speedup(module, latent_channels) and wide_enough:
            return FactorizedConv2d.from_conv2d(module, module_index, latent_channels=latent_channels)
        return None  # not enough rank reduction to be worth it

    return _replace_module_class_in_model(model, torch.nn.Conv2d, _maybe_replace_conv2d)


def factorize_linear_modules(model: torch.nn.Module, min_features: int, latent_features: FractionOrInt):

    def _maybe_replace_linear(module: torch.nn.Linear,
                              module_index: int,
                              min_features: int = min_features,
                              latent_features: FractionOrInt = latent_features) -> Optional[torch.nn.Module]:
        wide_enough = min(module.in_features, module.out_features) >= min_features
        if factorizing_could_speedup(module, latent_features) and wide_enough:
            return FactorizedLinear.from_linear(module, module_index, latent_features=latent_features)
        return None  # not enough rank reduction to be worth it

    return _replace_module_class_in_model(model, torch.nn.Linear, _maybe_replace_linear)


@dataclass
class FactorizeHparams(AlgorithmHparams):
    factorize_convs: bool = hp.optional(
        doc='Whether to factorize convolutional layers',
        default=_DEFAULT_SHOULD_FACTORIZE_CONVS,
    )
    factorize_linears: bool = hp.optional(
        doc='Whether to factorize linear layers',
        default=_DEFAULT_SHOULD_FACTORIZE_LINEARS,
    )
    min_channels: int = hp.optional(
        doc='Minimum number of channels in a Conv2d module'
        ' for it to be factorized.',
        default=_DEFAULT_MIN_CHANNELS,
    )
    latent_channels: float = hp.optional(
        doc='Number of channels in factorized convolution latent representations',
        default=_DEFAULT_LATENT_CHANNELS,
    )
    min_features: int = hp.optional(
        doc='Minimum number of features in a Linear module'
        ' for it to be factorized.',
        default=_DEFAULT_MIN_FEATURES,
    )
    latent_features: float = hp.optional(
        doc='Number of features in factorized linear latent representations',
        default=_DEFAULT_LATENT_FEATURES,
    )

    def initialize_object(self) -> 'Factorize':
        return Factorize(**asdict(self))


class Factorize(Algorithm):

    def __init__(self,
                 factorize_convs: bool = _DEFAULT_SHOULD_FACTORIZE_CONVS,
                 factorize_linears: bool = _DEFAULT_SHOULD_FACTORIZE_LINEARS,
                 min_channels: int = _DEFAULT_MIN_CHANNELS,
                 latent_channels: int = _DEFAULT_LATENT_CHANNELS,
                 min_features: int = _DEFAULT_MIN_FEATURES,
                 latent_features: int = _DEFAULT_LATENT_FEATURES):
        self.hparams = FactorizeHparams(factorize_convs=factorize_convs,
                                        factorize_linears=factorize_linears,
                                        min_channels=min_channels,
                                        latent_channels=latent_channels,
                                        min_features=min_features,
                                        latent_features=latent_features)

    def match(self, event: Event, state: State) -> bool:
        """Run on Event.INIT

        Args:
            event (:class:`Event`): The current event.
            state (:class:`State`): The current state.

        Returns:
            bool: True if this algorithm should run
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
        if self.hparams.factorize_convs:
            factorize_conv2d_modules(state.model,
                                     min_channels=self.hparams.min_channels,
                                     latent_channels=self.hparams.latent_channels)
            num_factorized = surgery.count_module_instances(state.model, FactorizedConv2d)
            logger.metric_fit({
                LOG_NUM_CONV2D_REPLACEMENTS_KEY: num_factorized,
            })
        if self.hparams.factorize_linears:
            factorize_linear_modules(state.model,
                                     min_features=self.hparams.min_features,
                                     latent_features=self.hparams.latent_features)
            num_factorized = surgery.count_module_instances(state.model, FactorizedLinear)
            logger.metric_fit({
                LOG_NUM_LINEAR_REPLACEMENTS_KEY: num_factorized,
            })
