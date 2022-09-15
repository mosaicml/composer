# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Modules and layers for applying the Stochastic Depth algorithm."""

from __future__ import annotations

import functools
import logging
from typing import Optional, Type, Union

import torch
from torchvision.models.resnet import Bottleneck

from composer.algorithms.stochastic_depth.stochastic_layers import make_resnet_bottleneck_stochastic
from composer.core import Algorithm, Event, State
from composer.core.time import Time, TimeUnit
from composer.loggers import Logger
from composer.utils import module_surgery

log = logging.getLogger(__name__)

_VALID_LAYER_DISTRIBUTIONS = ('uniform', 'linear')

_VALID_STOCHASTIC_METHODS = ('block', 'sample')

_STOCHASTIC_LAYER_MAPPING = {'ResNetBottleneck': (Bottleneck, make_resnet_bottleneck_stochastic)}

__all__ = ['apply_stochastic_depth', 'StochasticDepth']


def apply_stochastic_depth(model: torch.nn.Module,
                           target_layer_name: str,
                           stochastic_method: str = 'block',
                           drop_rate: float = 0.2,
                           drop_distribution: str = 'linear') -> torch.nn.Module:
    """Applies Stochastic Depth (`Huang et al, 2016 <https://arxiv.org/abs/1603.09382>`_) to the specified model.

    The algorithm replaces the specified target layer with a stochastic version
    of the layer. The stochastic layer will randomly drop either samples or the
    layer itself depending on the stochastic method specified. The block-wise
    version follows the original paper. The sample-wise version follows the
    implementation used for EfficientNet in the
    `Tensorflow/TPU repo <https://github.com/tensorflow/tpu>`_.

    .. note::

        Stochastic Depth only works on instances of :class:`torchvision.models.resnet.ResNet`
        for now.

    Args:
        model (torch.nn.Module): model containing modules to be replaced with
            stochastic versions.
        target_layer_name (str): Block to replace with a stochastic block
            equivalent. The name must be registered in ``STOCHASTIC_LAYER_MAPPING``
            dictionary with the target layer class and the stochastic layer class.
            Currently, only :class:`torchvision.models.resnet.Bottleneck` is supported.
        stochastic_method (str, optional): The version of stochastic depth to use.
            ``"block"`` randomly drops blocks during training. ``"sample"`` randomly
            drops samples within a block during training. Default: ``"block"``.
        drop_rate (float, optional): The base probability of dropping a layer or sample.
            Must be between 0.0 and 1.0. Default: `0.2``.
        drop_distribution (str, optional): How ``drop_rate`` is distributed across
            layers. Value must be one of ``"uniform"`` or ``"linear"``.
            ``"uniform"`` assigns the same ``drop_rate`` across all layers.
            ``"linear"`` linearly increases the drop rate across layer depth,
            starting with 0 drop rate and ending with ``drop_rate``.
            Default: ``"linear"``.

    Returns:
        The modified model

    Example:
        .. testcode::

            import composer.functional as cf
            from torchvision import models
            model = models.resnet50()
            cf.apply_stochastic_depth(
                model,
                target_layer_name='ResNetBottleneck'
            )
    """
    _validate_stochastic_hparams(target_layer_name=target_layer_name,
                                 stochastic_method=stochastic_method,
                                 drop_rate=drop_rate,
                                 drop_distribution=drop_distribution)
    transforms = {}
    target_layer, stochastic_converter = _STOCHASTIC_LAYER_MAPPING[target_layer_name]
    module_count = module_surgery.count_module_instances(model, target_layer)
    stochastic_from_target_layer = functools.partial(stochastic_converter,
                                                     drop_rate=drop_rate,
                                                     drop_distribution=drop_distribution,
                                                     module_count=module_count,
                                                     stochastic_method=stochastic_method)
    transforms[target_layer] = stochastic_from_target_layer
    module_surgery.replace_module_classes(model, policies=transforms)
    return model


class StochasticDepth(Algorithm):
    """Applies Stochastic Depth (`Huang et al, 2016 <https://arxiv.org/abs/1603.09382>`_) to the specified model.

    The algorithm replaces the specified target layer with a stochastic version
    of the layer. The stochastic layer will randomly drop either samples or the
    layer itself depending on the stochastic method specified. The block-wise
    version follows the original paper. The sample-wise version follows the
    implementation used for EfficientNet in the
    `Tensorflow/TPU repo <https://github.com/tensorflow/tpu>`_.

    Runs on :attr:`.Event.INIT`, as well as
    :attr:`.Event.BATCH_START` if ``drop_warmup > 0``.

    .. note::

        Stochastic Depth only works on instances of :class:`torchvision.models.resnet.ResNet` for now.

    Args:
        target_layer_name (str): Block to replace with a stochastic block
            equivalent. The name must be registered in ``STOCHASTIC_LAYER_MAPPING``
            dictionary with the target layer class and the stochastic layer class.
            Currently, only :class:`torchvision.models.resnet.Bottleneck` is supported.
        stochastic_method (str, optional): The version of stochastic depth to use.
            ``"block"`` randomly drops blocks during training. ``"sample"`` randomly drops
            samples within a block during training. Default: ``"block"``.
        drop_rate (float, optional): The base probability of dropping a layer or sample.
            Must be between 0.0 and 1.0. Default: ``0.2``.
        drop_distribution (str, optional): How ``drop_rate`` is distributed across
            layers. Value must be one of ``"uniform"`` or ``"linear"``.
            ``"uniform"`` assigns the same ``drop_rate`` across all layers.
            ``"linear"`` linearly increases the drop rate across layer depth,
            starting with 0 drop rate and ending with ``drop_rate``. Default: ``"linear"``.
        drop_warmup (str | Time | float, optional): A :class:`Time` object,
            time-string, or float on ``[0.0, 1.0]`` representing the fraction of the
            training duration to linearly increase the drop probability to
            `linear_drop_rate`. Default: ``0.0``.
    """

    def __init__(self,
                 target_layer_name: str,
                 stochastic_method: str = 'block',
                 drop_rate: float = 0.2,
                 drop_distribution: str = 'linear',
                 drop_warmup: Union[float, Time, str] = 0.0):

        if drop_rate == 0.0:
            log.warning('Stochastic Depth will have no effect when drop_rate set to 0')

        self.target_layer_name = target_layer_name
        self.stochastic_method = stochastic_method
        self.drop_rate = drop_rate
        self.drop_distribution = drop_distribution
        if isinstance(drop_warmup, str):
            drop_warmup = Time.from_timestring(drop_warmup)
        if isinstance(drop_warmup, float):
            drop_warmup = Time(drop_warmup, TimeUnit.DURATION)
        self.drop_warmup = drop_warmup
        self.num_stochastic_layers = 0  # Initial count of stochastic layers
        _validate_stochastic_hparams(stochastic_method=self.stochastic_method,
                                     target_layer_name=self.target_layer_name,
                                     drop_rate=self.drop_rate,
                                     drop_distribution=self.drop_distribution,
                                     drop_warmup=str(self.drop_warmup))

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(target_layer_name='{self.target_layer_name}',stochastic_method='{self.stochastic_method}',drop_rate={self.drop_rate},drop_distribution='{self.drop_distribution}',drop_warmup={repr(self.drop_warmup)})"

    @property
    def find_unused_parameters(self) -> bool:
        return self.stochastic_method == 'block'

    @staticmethod
    def required_on_load() -> bool:
        return True

    def match(self, event: Event, state: State) -> bool:
        return (event == Event.INIT) or (event == Event.BATCH_START and self.drop_warmup > 0.0)

    def apply(self, event: Event, state: State, logger: Logger) -> Optional[int]:
        assert state.model is not None
        target_block, _ = _STOCHASTIC_LAYER_MAPPING[self.target_layer_name]

        if event == Event.INIT:
            if module_surgery.count_module_instances(state.model, target_block) == 0:
                log.warning(f'No {self.target_layer_name} found in model! Algorithm will function as a no-op.')

            apply_stochastic_depth(state.model,
                                   target_layer_name=self.target_layer_name,
                                   stochastic_method=self.stochastic_method,
                                   drop_rate=self.drop_rate,
                                   drop_distribution=self.drop_distribution)
            self.num_stochastic_layers = module_surgery.count_module_instances(state.model, target_block)
            logger.log_metrics({'stochastic_depth/num_stochastic_layers': self.num_stochastic_layers})

        elif event == Event.BATCH_START and self.num_stochastic_layers:
            elapsed_duration = state.get_elapsed_duration()
            assert elapsed_duration is not None, 'elapsed duration is set on BATCH_START'
            if elapsed_duration < self.drop_warmup:
                current_drop_rate = float(elapsed_duration / self.drop_warmup) * self.drop_rate
                _update_drop_rate(module=state.model,
                                  target_block=target_block,
                                  drop_rate=current_drop_rate,
                                  drop_distribution=self.drop_distribution,
                                  module_count=self.num_stochastic_layers)
            else:
                current_drop_rate = self.drop_rate
            logger.log_metrics({'stochastic_depth/drop_rate': current_drop_rate})


def _validate_stochastic_hparams(target_layer_name: str,
                                 stochastic_method: str,
                                 drop_rate: float,
                                 drop_distribution: str,
                                 drop_warmup: str = '0dur'):
    """Helper function to validate the Stochastic Depth hyperparameter values.
    """

    if stochastic_method and (stochastic_method not in _VALID_STOCHASTIC_METHODS):
        raise ValueError(f'stochastic_method {stochastic_method} is not supported.'
                         f' Must be one of {_VALID_STOCHASTIC_METHODS}')

    if target_layer_name and (target_layer_name not in _STOCHASTIC_LAYER_MAPPING):
        raise ValueError(f'target_layer_name {target_layer_name} is not supported with {stochastic_method}.'
                         f' Must be one of {list(_STOCHASTIC_LAYER_MAPPING.keys())}')

    if drop_rate and (drop_rate < 0 or drop_rate > 1):
        raise ValueError(f'drop_rate must be between 0 and 1: {drop_rate}')

    if drop_distribution and (drop_distribution not in _VALID_LAYER_DISTRIBUTIONS):
        raise ValueError(f'drop_distribution "{drop_distribution}" is'
                         f' not supported. Must be one of {list(_VALID_LAYER_DISTRIBUTIONS)}')

    if stochastic_method == 'sample' and Time.from_timestring(drop_warmup).value != 0:
        raise ValueError(f'drop_warmup can not be used with "sample" stochastic_method')


def _update_drop_rate(module: torch.nn.Module,
                      target_block: Type[torch.nn.Module],
                      drop_rate: float,
                      drop_distribution: str,
                      module_count: int,
                      module_id: int = 0):
    """Recursively updates a module's drop_rate attributes with a new value.
    """

    for child in module.children():
        if isinstance(child, target_block) and hasattr(child, 'drop_rate'):
            module_id += 1
            if drop_distribution == 'linear':
                current_drop_rate = (module_id / module_count) * drop_rate  # type: ignore
            else:
                current_drop_rate = drop_rate
            child.drop_rate = torch.tensor(current_drop_rate)
        module_id = _update_drop_rate(child, target_block, drop_rate, drop_distribution, module_count, module_id)
    return module_id
