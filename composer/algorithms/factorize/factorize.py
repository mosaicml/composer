# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from typing import Optional, Sequence, Type, Union, cast

import torch
from torch.optim import Optimizer

from composer.algorithms.factorize.factorize_modules import (FactorizedConv2d, FactorizedLinear,
                                                             factorizing_could_speedup)
from composer.core import Algorithm, Event, State
from composer.loggers import Logger
from composer.utils import module_surgery

log = logging.getLogger(__name__)

LOG_NUM_CONV2D_REPLACEMENTS_KEY = 'factorize/num_conv2d_replacements'
LOG_NUM_LINEAR_REPLACEMENTS_KEY = 'factorize/num_linear_replacements'


def apply_factorization(model: torch.nn.Module,
                        factorize_convs: bool = True,
                        factorize_linears: bool = True,
                        min_channels: int = 512,
                        latent_channels: Union[int, float] = 0.25,
                        min_features: int = 512,
                        latent_features: Union[int, float] = 0.25,
                        optimizers: Optional[Union[Optimizer, Sequence[Optimizer]]] = None) -> torch.nn.Module:
    """Replaces :class:`torch.nn.Linear` and :class:`torch.nn.Conv2d` modules with
    :class:`.FactorizedLinear` and :class:`.FactorizedConv2d` modules.

    Factorized modules replace one full-rank operation with a sequence of two
    lower-rank operations. When the rank is low enough, this can save
    computation, at the cost of expressive power. See :class:`.Factorize` for details.

    Args:
        model (torch.nn.Module): the model to modify in-place.
        factorize_convs (bool, optional): whether to try factorizing :class:`torch.nn.Conv2d` modules.
            Default: ``True``.
        factorize_linears (bool, optional): whether to try factorizing :class:`torch.nn.Linear` modules.
            Default: ``True``.
        min_channels (int, optional): if a :class:`torch.nn.Conv2d` module does not have at least
            this many input and output channels, it will be ignored. Modules with
            few channels are unlikely to be accelerated by factorization due
            to poor hardware utilization. Default: ``512``.
        latent_channels (int | float, optional): number of latent channels to use in factorized
            convolutions. Can be specified as either an integer > 1 or as a
            float within ``[0, 1)``. In the latter case, the value is
            interpreted as a fraction of ``min(in_channels, out_channels)``
            for each :class:`torch.nn.Conv2d` module, and is converted to
            the equivalent integer value, with a minimum of 1. Default: ``0.25``.
        min_features (int, optional): if a :class:`torch.nn.Linear` module does not have at least
            this many input and output features, it will be ignored. Modules with
            few features are unlikely to be accelerated by factorization due
            to poor hardware utilization. Default: ``512``.
        latent_features (int | float, optional): size of the latent space for factorized linear modules.
            Can be specified as either an integer > 1 or as a float within ``[0, 0.5)``.
            In the latter case, the value is interpreted as a fraction of
            ``min(in_features, out_features)`` for each :class:`torch.nn.Linear`
            module, and is converted to the equivalent integer value, with a
            minimum of 1. Default: ``0.25``.
        optimizers (torch.optim.Optimizer | Sequence[torch.optim.Optimizer], optional):
            Existing optimizers bound to ``model.parameters()``. All optimizers that have already been
            constructed with ``model.parameters()`` must be specified here so
            that they will optimize the correct parameters.

            If the optimizer(s) are constructed *after* calling this function,
            then it is safe to omit this parameter. These optimizers will see
            the correct model parameters.

    Returns:
        The modified model

    Example:
        .. testcode::

            import composer.functional as cf
            from torchvision import models
            model = models.resnet50()
            cf.apply_factorization(model)
    """
    if factorize_convs:
        _factorize_conv2d_modules(model,
                                  min_channels=min_channels,
                                  latent_channels=latent_channels,
                                  optimizers=optimizers)
    if factorize_linears:
        _factorize_linear_modules(model,
                                  min_features=min_features,
                                  latent_features=latent_features,
                                  optimizers=optimizers)
    return model


class Factorize(Algorithm):
    """Decomposes linear operators into pairs of smaller linear operators.

    Specifically, this algorithm replaces :class:`torch.nn.Conv2d` and
    :class:`torch.nn.Linear` modules with :class:`.FactorizedConv2d` and
    :class:`.FactorizedLinear` modules.

    The replacement is only performed if doing so would reduce the number of
    multiply-adds used to compute each module's output. For linear
    layers and pointwise convolutions, this means that the factorization must
    use an intermediate rank of less than half the input and output ranks, since
    it must perform two operations instead of one.

    For convolutions with kernel sizes greater than 1, the threshold for
    factorization being worthwhile varies with kernel size. Larger kernels
    allow larger intermediate ranks.

    See :func:`.factorize_matrix` and :func:`.factorize_conv2d` for more
    information about the factorization process. See :class:`.FactorizedConv2d`
    and :class:`.FactorizedLinear` for more information about the factorized modules
    used to replace the original modules.

    Runs on :attr:`.Event.INIT`.

    Args:
        factorize_convs (bool): whether to try factorizing :class:`torch.nn.Conv2d` modules.
            Default: ``True``.
        factorize_linears (bool): whether to try factorizing :class:`torch.nn.Linear` modules.
            Default: ``True``.
        min_channels (int): if a :class:`torch.nn.Conv2d` module does not have at least
            this many input and output channels, it will be ignored. Modules with
            few channels are unlikely to be accelerated by factorization due
            to poor hardware utilization. Default: ``256``.
        latent_channels (int, float): number of latent channels to use in factorized
            convolutions. Can be specified as either an integer > 1 or as
            a float within ``[0, 1)``. In the latter case, the value is
            interpreted as a fraction of ``min(in_channels, out_channels)``
            for each :class:`torch.nn.Conv2d` module, and is converted to
            the equivalent integer value, with a minimum of 1. Default: ``0.25``.
        min_features (int): if a :class:`torch.nn.Linear` module does not have at least
            this many input and output features, it will be ignored. Modules with
            few features are unlikely to be accelerated by factorization due
            to poor hardware utilization. Default: ``256``.
        latent_features (int, float): size of the latent space for factorized linear modules.
            Can be specified as either an integer > 1 or as a float within ``[0, 0.5)``.
            In the latter case, the value is interpreted as a fraction of
            ``min(in_features, out_features)`` for each :class:`torch.nn.Linear`
            module and is converted to the equivalent integer value, with a
            minimum of 1. Default: ``128``.
    """

    def __init__(self,
                 factorize_convs: bool = True,
                 factorize_linears: bool = True,
                 min_channels: int = 256,
                 latent_channels: Union[int, float] = 0.25,
                 min_features: int = 256,
                 latent_features: Union[int, float] = 128):
        self.factorize_convs = factorize_convs
        self.factorize_linears = factorize_linears
        self.min_channels = min_channels
        self.latent_channels = latent_channels
        self.min_features = min_features
        self.latent_features = latent_features

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(factorize_convs={self.factorize_convs},factorize_linears={self.factorize_linears},min_channels={self.min_channels},latent_channels={self.latent_channels},min_features={self.min_features},latent_features={self.latent_features})'

    @staticmethod
    def required_on_load() -> bool:
        return True

    def match(self, event: Event, state: State) -> bool:
        return event == Event.INIT

    def apply(self, event: Event, state: State, logger: Logger) -> Optional[int]:
        assert state.model is not None, 'Model must be part of state!'

        apply_factorization(model=state.model,
                            factorize_convs=self.factorize_convs,
                            factorize_linears=self.factorize_linears,
                            min_channels=self.min_channels,
                            latent_channels=self.latent_channels,
                            min_features=self.min_features,
                            latent_features=self.latent_features,
                            optimizers=state.optimizers)

        if self.factorize_convs:
            num_factorized = module_surgery.count_module_instances(state.model, FactorizedConv2d)
            logger.log_hyperparameters({
                LOG_NUM_CONV2D_REPLACEMENTS_KEY: num_factorized,
            })
        if self.factorize_linears:
            num_factorized = module_surgery.count_module_instances(state.model, FactorizedLinear)
            logger.log_hyperparameters({
                LOG_NUM_LINEAR_REPLACEMENTS_KEY: num_factorized,
            })


def _python_log_surgery_result(model: torch.nn.Module, new_class: Type[torch.nn.Module]):
    num_replaced_modules = module_surgery.count_module_instances(model, new_class)
    log.info(f'Applied factorization to model {model.__class__.__name__}. ' +
             f'Model now has {num_replaced_modules} {new_class.__name__} modules')


def _factorize_conv2d_modules(model: torch.nn.Module,
                              min_channels: int = 512,
                              latent_channels: Union[int, float] = 0.25,
                              optimizers: Optional[Union[Optimizer, Sequence[Optimizer]]] = None):
    """Replaces :class:`torch.nn.Conv2d` modules in ``model`` with
    :class:`.FactorizedConv2d` modules.

    See :class:`.Factorize` for details.
    """

    def _maybe_replace_conv2d(module: torch.nn.Module, module_index: int) -> Optional[torch.nn.Module]:
        module = cast(torch.nn.Conv2d, module)
        wide_enough = min(module.out_channels, module.in_channels) >= min_channels
        if factorizing_could_speedup(module, latent_channels) and wide_enough:
            return FactorizedConv2d.from_conv2d(module, module_index, latent_channels=latent_channels)
        return None  # not enough rank reduction to be worth it

    ret = module_surgery.replace_module_classes(model,
                                                optimizers=optimizers,
                                                policies={torch.nn.Conv2d: _maybe_replace_conv2d})
    _python_log_surgery_result(model, FactorizedConv2d)
    return ret


def _factorize_linear_modules(model: torch.nn.Module,
                              min_features: int = 512,
                              latent_features: Union[int, float] = 0.25,
                              optimizers: Optional[Union[Optimizer, Sequence[Optimizer]]] = None):
    """Replaces :class:`torch.nn.Linear` modules in ``model`` with
    :class:`.FactorizedLinear` modules.

    See :class:`.Factorize` for details.
    """

    def _maybe_replace_linear(module: torch.nn.Module, module_index: int) -> Optional[torch.nn.Module]:
        module = cast(torch.nn.Linear, module)
        wide_enough = min(module.in_features, module.out_features) >= min_features
        if factorizing_could_speedup(module, latent_features) and wide_enough:
            return FactorizedLinear.from_linear(module, module_index, latent_features=latent_features)
        return None  # not enough rank reduction to be worth it

    ret = module_surgery.replace_module_classes(model,
                                                optimizers=optimizers,
                                                policies={torch.nn.Linear: _maybe_replace_linear})
    _python_log_surgery_result(model, FactorizedLinear)
    return ret
