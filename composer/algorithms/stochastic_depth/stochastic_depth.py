# Copyright 2021 MosaicML. All Rights Reserved.

from __future__ import annotations

import functools
import logging
from typing import Optional, Type, Union

import torch
from torchvision.models.resnet import Bottleneck

from composer.algorithms.stochastic_depth.sample_stochastic_layers import SampleStochasticBottleneck
from composer.algorithms.stochastic_depth.stochastic_layers import StochasticBottleneck
from composer.core import Algorithm, Event, Logger, State
from composer.core.time import Time, TimeUnit
from composer.core.types import Optimizers
from composer.utils import module_surgery

log = logging.getLogger(__name__)

_VALID_LAYER_DISTRIBUTIONS = ("uniform", "linear")

_STOCHASTIC_LAYER_MAPPING = {
    'block': {
        'ResNetBottleneck': (Bottleneck, StochasticBottleneck)
    },
    'sample': {
        'ResNetBottleneck': (Bottleneck, SampleStochasticBottleneck)
    }
}


def apply_stochastic_depth(model: torch.nn.Module,
                           target_layer_name: str,
                           stochastic_method: str = 'block',
                           drop_rate: float = 0.2,
                           drop_distribution: str = 'linear',
                           use_same_gpu_seed: bool = True,
                           optimizers: Optional[Optimizers] = None) -> torch.nn.Module:
    """Applies Stochastic Depth (`Huang et al, 2016 <https://arxiv.org/abs/1603.09382>`_) to the specified model.

    The algorithm replaces the specified target layer with a stochastic version
    of the layer. The stochastic layer will randomly drop either samples or the
    layer itself depending on the stochastic method specified. The block-wise
    version follows the original paper. The sample-wise version follows the
    implementation used for EfficientNet in the
    `Tensorflow/TPU repo <https://github.com/tensorflow/tpu>`_.

    .. note::

        Stochastic Depth only works on instances of `torchvision.models.resnet.ResNet` for now.

    Args:
        model (torch.nn.Module): model containing modules to be replaced with stochastic versions
        target_layer_name (str): Block to replace with a stochastic block
            equivalent. The name must be registered in ``STOCHASTIC_LAYER_MAPPING``
            dictionary with the target layer class and the stochastic layer class.
            Currently, only :class:`torchvision.models.resnet.Bottleneck` is supported.
        stochastic_method (str, optional): The version of stochastic depth to use. ``"block"``
            randomly drops blocks during training. ``"sample"`` randomly drops
            samples within a block during training. Default: ``"block"``.
        drop_rate (float, optional): The base probability of dropping a layer or sample. Must be
            between 0.0 and 1.0. Default: `0.2``.
        drop_distribution (str, optional): How ``drop_rate`` is distributed across
            layers. Value must be one of ``"uniform"`` or ``"linear"``.
            ``"uniform"`` assigns the same ``drop_rate`` across all layers.
            ``"linear"`` linearly increases the drop rate across layer depth
            starting with 0 drop rate and ending with ``drop_rate``. Default: ``"linear"``.
        use_same_gpu_seed (bool, optional): Set to ``True`` to have the same layers dropped
            across GPUs when using multi-GPU training. Set to ``False`` to
            have each GPU drop a different set of layers. Only used
            with ``"block"`` stochastic method. Default: ``True``.
        optimizers (Optimizers, optional):  Existing optimizers bound to ``model.parameters()``.
            All optimizers that have already been constructed with
            ``model.parameters()`` must be specified here so they will optimize
            the correct parameters.

            If the optimizer(s) are constructed *after* calling this function,
            then it is safe to omit this parameter. These optimizers will see the correct
            model parameters.

    Returns:
        The modified model

    Example:
        .. testcode::

            import composer.functional as cf
            from torchvision import models
            model = models.resnet50()
            cf.apply_stochastic_depth(model, target_layer_name='ResNetBottleneck')
    """
    _validate_stochastic_hparams(target_layer_name=target_layer_name,
                                 stochastic_method=stochastic_method,
                                 drop_rate=drop_rate,
                                 drop_distribution=drop_distribution)
    transforms = {}
    target_layer, stochastic_layer = _STOCHASTIC_LAYER_MAPPING[stochastic_method][target_layer_name]
    module_count = module_surgery.count_module_instances(model, target_layer)
    shared_kwargs = {'drop_rate': drop_rate, 'drop_distribution': drop_distribution, 'module_count': module_count}
    if stochastic_method == 'block':
        rand_generator = torch.Generator()  # Random number generator for each layer
        stochastic_from_target_layer = functools.partial(stochastic_layer.from_target_layer,
                                                         **shared_kwargs,
                                                         use_same_gpu_seed=use_same_gpu_seed,
                                                         rand_generator=rand_generator)
    elif stochastic_method == 'sample':
        stochastic_from_target_layer = functools.partial(stochastic_layer.from_target_layer, **shared_kwargs)
    else:
        raise ValueError(f"stochastic_method {stochastic_method} is not supported."
                         f" Must be one of {list(_STOCHASTIC_LAYER_MAPPING.keys())}")
    transforms[target_layer] = stochastic_from_target_layer
    module_surgery.replace_module_classes(model, optimizers=optimizers, policies=transforms)
    return model


class StochasticDepth(Algorithm):
    """Applies Stochastic Depth (`Huang et al, 2016 <https://arxiv.org/abs/1603.09382>`_) to the specified model.

    The algorithm replaces the specified target layer with a stochastic version
    of the layer. The stochastic layer will randomly drop either samples or the
    layer itself depending on the stochastic method specified. The block-wise
    version follows the original paper. The sample-wise version follows the
    implementation used for EfficientNet in the
    `Tensorflow/TPU repo <https://github.com/tensorflow/tpu>`_.

    Runs on :attr:`~composer.core.event.Event.INIT`, as well as
    :attr:`~composer.core.event.Event.BATCH_START` if ``drop_warmup > 0``.

    .. note::

        Stochastic Depth only works on instances of `torchvision.models.resnet.ResNet` for now.

    Args:
        target_layer_name (str): Block to replace with a stochastic block
            equivalent. The name must be registered in ``STOCHASTIC_LAYER_MAPPING``
            dictionary with the target layer class and the stochastic layer class.
            Currently, only :class:`torchvision.models.resnet.Bottleneck` is supported.
        stochastic_method (str, optional): The version of stochastic depth to use. ``"block"``
            randomly drops blocks during training. ``"sample"`` randomly drops
            samples within a block during training. Default: ``"block"``.
        drop_rate (float, optional): The base probability of dropping a layer or sample. Must be
            between 0.0 and 1.0. Default: ``0.2``.
        drop_distribution (str, optional): How ``drop_rate`` is distributed across
            layers. Value must be one of ``"uniform"`` or ``"linear"``.
            ``"uniform"`` assigns the same ``drop_rate`` across all layers.
            ``"linear"`` linearly increases the drop rate across layer depth
            starting with 0 drop rate and ending with ``drop_rate``. Default: ``"linear"``.
        drop_warmup (str | Time | float, optional): A :class:`Time` object, time-string, or float
            on [0.0; 1.0] representing the fraction of the training duration to linearly
            increase the drop probability to `linear_drop_rate`. Default: ``0.0``.
        use_same_gpu_seed (bool, optional): Set to ``True`` to have the same layers dropped
            across GPUs when using multi-GPU training. Set to ``False`` to
            have each GPU drop a different set of layers. Only used
            with ``"block"`` stochastic method. Default: ``True``.
    """

    def __init__(self,
                 target_layer_name: str,
                 stochastic_method: str = 'block',
                 drop_rate: float = 0.2,
                 drop_distribution: str = 'linear',
                 drop_warmup: Union[float, Time, str] = 0.0,
                 use_same_gpu_seed: bool = True):

        if drop_rate == 0.0:
            log.warning('Stochastic Depth will have no effect when drop_rate set to 0')

        if stochastic_method == "sample" and not use_same_gpu_seed:
            log.warning('use_same_gpu_seed=false has no effect when using the "sample" method')

        self.target_layer_name = target_layer_name
        self.stochastic_method = stochastic_method
        self.drop_rate = drop_rate
        self.drop_distribution = drop_distribution
        if isinstance(drop_warmup, str):
            drop_warmup = Time.from_timestring(drop_warmup)
        if isinstance(drop_warmup, float):
            drop_warmup = Time(drop_warmup, TimeUnit.DURATION)
        self.drop_warmup = drop_warmup
        self.use_same_gpu_seed = use_same_gpu_seed
        _validate_stochastic_hparams(stochastic_method=self.stochastic_method,
                                     target_layer_name=self.target_layer_name,
                                     drop_rate=self.drop_rate,
                                     drop_distribution=self.drop_distribution,
                                     drop_warmup=str(self.drop_warmup))

    @property
    def find_unused_parameters(self) -> bool:
        """DDP parameter to notify that parameters may not have gradients if it is dropped during the forward pass."""

        return (self.stochastic_method == "block")

    def match(self, event: Event, state: State) -> bool:
        """Run on :attr:`~composer.core.event.Event.INIT`, as well as
    :attr:`~composer.core.event.Event.BATCH_START` if ``drop_warmup > 0``.

        Args:
            event (Event): The current event.
            state (State): The current state.
        Returns:
            bool: True if this algorithm should run now.
        """

        return (event == Event.INIT) or (event == Event.BATCH_START and self.drop_warmup > 0.0)

    def apply(self, event: Event, state: State, logger: Logger) -> Optional[int]:
        """Applies StochasticDepth modification to the state's model.

        Args:
            event (Event): the current event
            state (State): the current trainer state
            logger (Logger): the training logger
        """
        assert state.model is not None
        target_layer, stochastic_layer = _STOCHASTIC_LAYER_MAPPING[self.stochastic_method][self.target_layer_name]

        if event == Event.INIT:
            if module_surgery.count_module_instances(state.model, target_layer) == 0:
                log.warning(f'No {self.target_layer_name} found in model! Algorithm will function as a no-op.')

            apply_stochastic_depth(state.model,
                                   optimizers=state.optimizers,
                                   target_layer_name=self.target_layer_name,
                                   stochastic_method=self.stochastic_method,
                                   drop_rate=self.drop_rate,
                                   drop_distribution=self.drop_distribution,
                                   use_same_gpu_seed=self.use_same_gpu_seed)
            num_stochastic_layers = module_surgery.count_module_instances(state.model, stochastic_layer)
            logger.metric_epoch({'stochastic_depth/num_stochastic_layers': num_stochastic_layers})

        elif event == Event.BATCH_START:
            if state.get_elapsed_duration() < self.drop_warmup:
                current_drop_rate = float(state.get_elapsed_duration() / self.drop_warmup) * self.drop_rate
                _update_drop_rate(state.model, stochastic_layer, current_drop_rate, self.drop_distribution)
            else:
                current_drop_rate = self.drop_rate
            logger.metric_batch({'stochastic_depth/drop_rate': current_drop_rate})


def _validate_stochastic_hparams(target_layer_name: str,
                                 stochastic_method: str,
                                 drop_rate: float,
                                 drop_distribution: str,
                                 drop_warmup: str = "0dur"):
    """Helper function to validate the Stochastic Depth hyperparameter values."""

    if stochastic_method and (stochastic_method not in _STOCHASTIC_LAYER_MAPPING):
        raise ValueError(f"stochastic_method {stochastic_method} is not supported."
                         f" Must be one of {list(_STOCHASTIC_LAYER_MAPPING.keys())}")

    if target_layer_name and (target_layer_name not in _STOCHASTIC_LAYER_MAPPING[stochastic_method]):
        raise ValueError(f"target_layer_name {target_layer_name} is not supported with {stochastic_method}."
                         f" Must be one of {list(_STOCHASTIC_LAYER_MAPPING[stochastic_method].keys())}")

    if drop_rate and (drop_rate < 0 or drop_rate > 1):
        raise ValueError(f"drop_rate must be between 0 and 1: {drop_rate}")

    if drop_distribution and (drop_distribution not in _VALID_LAYER_DISTRIBUTIONS):
        raise ValueError(f"drop_distribution '{drop_distribution}' is"
                         f" not supported. Must be one of {list(_VALID_LAYER_DISTRIBUTIONS)}")

    if stochastic_method == "sample" and Time.from_timestring(drop_warmup).value != 0:
        raise ValueError(f"drop_warmup can not be used with 'sample' stochastic_method")


def _update_drop_rate(module: torch.nn.Module, stochastic_block: Type[torch.nn.Module], drop_rate: float,
                      drop_distribution: str):
    """Recursively updates a module's drop_rate attributes with a new value."""

    if (len(list(module.children())) == 0 and len(list(module.parameters())) > 0):
        return
    else:
        for child in module.children():
            if isinstance(child, stochastic_block):
                if drop_distribution == 'uniform':
                    current_drop_rate = drop_rate
                elif drop_distribution == 'linear':
                    current_drop_rate = ((child.module_id + 1) / child.module_count) * drop_rate  # type: ignore
                else:
                    raise ValueError(f"drop_distribution '{drop_distribution}' is"
                                     f" not supported. Must be one of {list(_VALID_LAYER_DISTRIBUTIONS)}")
                child.drop_rate = torch.tensor(current_drop_rate)
            _update_drop_rate(child, stochastic_block, drop_rate, drop_distribution)
