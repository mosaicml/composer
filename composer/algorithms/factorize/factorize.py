# Copyright 2021 MosaicML. All Rights Reserved.

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Optional, Type, Union, cast

import torch
import yahp as hp

from composer.algorithms import AlgorithmHparams
from composer.algorithms.factorize.factorize_modules import (FactorizedConv2d, FactorizedLinear,
                                                             factorizing_could_speedup)
from composer.core import Algorithm, Event, Logger, State, surgery
from composer.core.types import Optimizers

log = logging.getLogger(__name__)

LOG_NUM_CONV2D_REPLACEMENTS_KEY = 'factorize/num_conv2d_replacements'
LOG_NUM_LINEAR_REPLACEMENTS_KEY = 'factorize/num_linear_replacements'


def _python_log_surgery_result(model: torch.nn.Module, new_class: Type[torch.nn.Module]):
    num_replaced_modules = surgery.count_module_instances(model, new_class)
    log.info(f'Applied factorization to model {model.__class__.__name__}. ' +
             f'Model now has {num_replaced_modules} {new_class.__name__} modules')


def factorize_conv2d_modules(model: torch.nn.Module,
                             min_channels: int,
                             latent_channels: Union[int, float],
                             optimizers: Optional[Optimizers] = None):
    """Replaces :class:`torch.nn.Conv2d` modules in ``model`` with :class:`~composer.algorithms.factorize.FactorizedConv2d` modules. See :class:`Factorize` for details."""

    def _maybe_replace_conv2d(module: torch.nn.Module, module_index: int) -> Optional[torch.nn.Module]:
        module = cast(torch.nn.Conv2d, module)
        wide_enough = min(module.out_channels, module.in_channels) >= min_channels
        if factorizing_could_speedup(module, latent_channels) and wide_enough:
            return FactorizedConv2d.from_conv2d(module, module_index, latent_channels=latent_channels)
        return None  # not enough rank reduction to be worth it

    ret = surgery.replace_module_classes(model,
                                         optimizers=optimizers,
                                         policies={torch.nn.Conv2d: _maybe_replace_conv2d})
    _python_log_surgery_result(model, FactorizedConv2d)
    return ret


def factorize_linear_modules(model: torch.nn.Module,
                             min_features: int,
                             latent_features: Union[int, float],
                             optimizers: Optional[Optimizers] = None):
    """Replaces :class:`torch.nn.Linear` modules in ``model`` with :class:`~composer.algorithms.factorize.FactorizedLinear` modules. See :class:`Factorize` for details."""

    def _maybe_replace_linear(module: torch.nn.Module, module_index: int) -> Optional[torch.nn.Module]:
        module = cast(torch.nn.Linear, module)
        wide_enough = min(module.in_features, module.out_features) >= min_features
        if factorizing_could_speedup(module, latent_features) and wide_enough:
            return FactorizedLinear.from_linear(module, module_index, latent_features=latent_features)
        return None  # not enough rank reduction to be worth it

    ret = surgery.replace_module_classes(model,
                                         optimizers=optimizers,
                                         policies={torch.nn.Linear: _maybe_replace_linear})
    _python_log_surgery_result(model, FactorizedLinear)
    return ret


@dataclass
class FactorizeHparams(AlgorithmHparams):
    """See :class:`Factorize`"""
    factorize_convs: bool = hp.optional(
        doc='Whether to factorize convolutional layers',
        default=True,
    )
    factorize_linears: bool = hp.optional(
        doc='Whether to factorize linear layers',
        default=True,
    )
    min_channels: int = hp.optional(
        doc=('Minimum number of channels in a Conv2d module' + ' for it to be factorized.'),
        default=256,
    )
    latent_channels: float = hp.optional(
        doc='Number of channels in factorized convolution latent representations',
        default=128,
    )
    min_features: int = hp.optional(
        doc=('Minimum number of features in a Linear module' + ' for it to be factorized.'),
        default=256,
    )
    latent_features: float = hp.optional(
        doc='Number of features in factorized linear latent representations',
        default=128,
    )

    def initialize_object(self) -> Factorize:
        return Factorize(**asdict(self))


class Factorize(Algorithm):
    """Decomposes linear operators into pairs of smaller linear operators.

    Specifically, this algorithm replaces :class:`~torch.nn.Conv2d` and
    :class:`~torch.nn.Linear` modules with
    :class:`~composer.algorithms.factorize.FactorizedConv2d` and
    :class:`~composer.algorithms.factorize.FactorizedLinear` modules.

    The replacement is only performed if doing so would reduce the number of
    multiply-adds used to compute each module's output. For linear
    layers and pointwise convolutions, this means that the factorization must
    use an intermediate rank of less than half the input and output ranks, since
    it must perform two operations instead of one.

    For convolutions with kernel sizes greater than 1, the threshold for
    factorization being worthwhile varies with kernel size. Larger kernels
    have lower thresholds.

    See :func:`~composer.algorithms.factorize.factorize_matrix` and
    :func:`~composer.algorithms.factorize.factorize_conv2d` for more
    information about the factorization process. See :class:`~composer.algorithms.factorize.FactorizedConv2d` and :class:`~composer.algorithms.factorize.FactorizedLinear`
    for more information about the factorized modules used to replace the
    original modules.

    Args:
        factorize_convs: whether to try factorizing :class:`torch.nn.Conv2d` modules.
        factorize_linears: whether to try factorizing :class:`torch.nn.Linear` modules.
        min_channels: if a :class:`~torch.nn.Conv2d` module does not have at least
            this many input and output channels, it will be ignored. Modules with
            few channels are unlikely to be accelerated by factorization due
            to poor hardware utilization.
        latent_channels: number of latent channels to use in factorized
            convolutions. Can be specified as either an integer > 1 or as
            float within [0, 1). In the latter case, the value is
            interpreted as a fraction of ``min(in_channels, out_channels)``
            for each :class:`~torch.nn.Conv2d` module, and is converted to
            the equivalent integer value, with a minimum of 1.
        min_features: if a :class:`~torch.nn.Linear` module does not have at least
            this many input and output features, it will be ignored. Modules with
            few features are unlikely to be accelerated by factorization due
            to poor hardware utilization.
        latent_features: size of the latent space for factorized linear modules.
            Can be specified as either an integer > 1 or as a float within [0, 0.5).
            In the latter case, the value is interpreted as a fraction of
            ``min(in_features, out_features)`` for each :class:`~torch.nn.Linear`
            module, and is converted to the equivalent integer value, with a
            minimum of 1.
    """

    def __init__(self,
                 factorize_convs: bool = True,
                 factorize_linears: bool = True,
                 min_channels: int = 256,
                 latent_channels: Union[int, float] = 128,
                 min_features: int = 256,
                 latent_features: Union[int, float] = 128):
        self.factorize_convs = factorize_convs
        self.factorize_linears = factorize_linears
        self.min_channels = min_channels
        self.latent_channels = latent_channels
        self.min_features = min_features
        self.latent_features = latent_features

    def match(self, event: Event, state: State) -> bool:
        """Run on Event.INIT

        Args:
            event: The current event.
            state: The current state.

        Returns:
            bool: True if this algorithm should run
        """
        return event == Event.INIT

    def apply(self, event: Event, state: State, logger: Logger) -> Optional[int]:
        """Factorize convolutional and linear layers

        Args:
            event: the current event
            state: the current trainer state
            logger: the training logger
        """
        assert state.model is not None, "Model must be part of state!"
        if self.factorize_convs:
            factorize_conv2d_modules(state.model,
                                     min_channels=self.min_channels,
                                     latent_channels=self.latent_channels,
                                     optimizers=state.optimizers)
            num_factorized = surgery.count_module_instances(state.model, FactorizedConv2d)
            logger.metric_fit({
                LOG_NUM_CONV2D_REPLACEMENTS_KEY: num_factorized,
            })
        if self.factorize_linears:
            factorize_linear_modules(state.model,
                                     min_features=self.min_features,
                                     latent_features=self.latent_features,
                                     optimizers=state.optimizers)
            num_factorized = surgery.count_module_instances(state.model, FactorizedLinear)
            logger.metric_fit({
                LOG_NUM_LINEAR_REPLACEMENTS_KEY: num_factorized,
            })
