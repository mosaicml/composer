# Copyright 2021 MosaicML. All Rights Reserved.

from __future__ import annotations

import functools
import logging
from dataclasses import asdict, dataclass
from typing import Optional, Type

import torch
import yahp as hp

from composer.algorithms import AlgorithmHparams
from composer.algorithms.stochastic_depth.sample_stochastic_layers import SampleStochasticBottleneck
from composer.algorithms.stochastic_depth.stochastic_layers import StochasticBottleneck
from composer.core import Algorithm, Event, Logger, State, surgery
from composer.models.resnets import Bottleneck

log = logging.getLogger(__name__)

_VALID_LAYER_DISTRIBUTIONS = ("uniform", "linear")

STOCHASTIC_LAYER_MAPPING = {
    'block': {
        'ResNetBottleneck': (Bottleneck, StochasticBottleneck)
    },
    'sample': {
        'ResNetBottleneck': (Bottleneck, SampleStochasticBottleneck)
    }
}


def validate_stochastic_hparams(stochastic_method: str,
                                target_layer_name: str,
                                drop_rate: float,
                                drop_distribution: str,
                                drop_warmup: float = 0.0):
    """Helper function to validate the Stochastic Depth hyperparameter values"""

    if stochastic_method and (stochastic_method not in STOCHASTIC_LAYER_MAPPING):
        raise ValueError(f"stochastic_method {stochastic_method} is not supported."
                         f" Must be one of {list(STOCHASTIC_LAYER_MAPPING.keys())}")

    if target_layer_name and (target_layer_name not in STOCHASTIC_LAYER_MAPPING[stochastic_method]):
        raise ValueError(f"target_layer_name {target_layer_name} is not supported with {stochastic_method}."
                         f" Must be one of {list(STOCHASTIC_LAYER_MAPPING[stochastic_method].keys())}")

    if drop_rate and (drop_rate < 0 or drop_rate > 1):
        raise ValueError(f"drop_rate must be between 0 and 1: {drop_rate}")

    if drop_distribution and (drop_distribution not in _VALID_LAYER_DISTRIBUTIONS):
        raise ValueError(f"drop_distribution '{drop_distribution}' is"
                         f" not supported. Must be one of {list(_VALID_LAYER_DISTRIBUTIONS)}")

    if drop_warmup and (drop_warmup < 0 or drop_warmup > 1):
        raise ValueError(f"drop_warmup must be between 0 and 1, not {drop_warmup}")

    if stochastic_method == "sample" and drop_warmup > 0:
        raise ValueError(f"drop_warmup can not be used with 'sample' stochastic_method")


@dataclass
class StochasticDepthHparams(AlgorithmHparams):
    """See :class:`StochasticDepth`"""

    stochastic_method: str = hp.required('The version of stochastic depth to use. One of ["sample", "block"].')
    target_layer_name: str = hp.required(
        f'Reference name of layer to replace. "block" method can be {list(STOCHASTIC_LAYER_MAPPING["block"].keys())}.'
        f' "sample" method can be {list(STOCHASTIC_LAYER_MAPPING["sample"].keys())}.')
    drop_rate: float = hp.optional('The probability of dropping a block or sample.', default=0.2)
    drop_distribution: str = hp.optional(
        '"Uniform" keeps the drop rate the same across blocks. "linear" linearly'
        ' increases the drop rate with block depth until it reaches `drop_rate`.',
        default='linear')
    use_same_gpu_seed: bool = hp.optional(
        'Whether or not to drop the same blocks across GPUs. Only used with "block" method.', default=True)
    drop_warmup: float = hp.optional(
        'Percentage of training to warmup `drop_rate`. Only use with "block" stochastic method.', default=0.0)

    def initialize_object(self) -> StochasticDepth:
        return StochasticDepth(**asdict(self))

    def validate(self):
        super().validate()
        validate_stochastic_hparams(stochastic_method=self.stochastic_method,
                                    target_layer_name=self.target_layer_name,
                                    drop_rate=self.drop_rate,
                                    drop_distribution=self.drop_distribution,
                                    drop_warmup=self.drop_warmup)


def apply_stochastic_depth(model: torch.nn.Module,
                           stochastic_method: str,
                           target_layer_name: str,
                           drop_rate: float = 0.2,
                           drop_distribution: str = 'linear',
                           use_same_gpu_seed: bool = True) -> None:
    """Applies Stochastic Depth (`Huang et al. <https://arxiv.org/abs/1603.09382>`_) to the specified model.

    The algorithm replaces the specified target layer with a stochastic version
    of the layer. The stochastic layer will randomly drop either samples or the
    layer itself depending on the stochastic method specified. The block-wise
    version follows the original paper. The sample-wise version follows the
    implementation used for EfficientNet in the
    `Tensorflow/TPU repo <https://github.com/tensorflow/tpu>`_.

    Args:
        model: model containing modules to be replaced with stochastic versions
        stochastic_method: The version of stochastic depth to use. ``"block"``
            randomly drops blocks during training. ``"sample"`` randomly drops
            samples within a block during training.
        target_layer_name: Block to replace with a stochastic block
            equivalent. The name must be registered in ``STOCHASTIC_LAYER_MAPPING``
            dictionary with the target layer class and the stochastic layer class.
            Currently, only ``'ResNetBottleneck'`` is supported.
        drop_rate: The base probability of dropping a layer or sample. Must be
            between 0.0 and 1.0.
        drop_distribution: How ``drop_rate`` is distributed across
            layers. Value must be one of ``"uniform"`` or ``"linear"``.
            ``"uniform"`` assigns the same ``drop_rate`` across all layers.
            ``"linear"`` linearly increases the drop rate across layer depth
            starting with 0 drop rate and ending with ``drop_rate``.
        use_same_gpu_seed: Set to ``True`` to have the same layers dropped
            across GPUs when using multi-GPU training. Set to ``False`` to
            have each GPU drop a different set of layers. Only used
            with ``"block"`` stochastic method.
    """
    validate_stochastic_hparams(stochastic_method=stochastic_method,
                                target_layer_name=target_layer_name,
                                drop_rate=drop_rate,
                                drop_distribution=drop_distribution)
    transforms = {}
    target_layer, stochastic_layer = STOCHASTIC_LAYER_MAPPING[stochastic_method][target_layer_name]
    module_count = surgery.count_module_instances(model, target_layer)
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
                         f" Must be one of {list(STOCHASTIC_LAYER_MAPPING.keys())}")
    transforms[target_layer] = stochastic_from_target_layer
    surgery.replace_module_classes(model, policies=transforms)


def _update_drop_rate(module: torch.nn.Module, stochastic_block: Type[torch.nn.Module], drop_rate: float,
                      drop_distribution: str):
    """Recursively updates a module's drop_rate attributes with a new value"""

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


class StochasticDepth(Algorithm):
    """Applies Stochastic Depth (`Huang et al. <https://arxiv.org/abs/1603.09382>`_) to the specified model.

    The algorithm replaces the specified target layer with a stochastic version
    of the layer. The stochastic layer will randomly drop either samples or the
    layer itself depending on the stochastic method specified. The block-wise
    version follows the original paper. The sample-wise version follows the
    implementation used for EfficientNet in the
    `Tensorflow/TPU repo <https://github.com/tensorflow/tpu>`_.

    Args:
        stochastic_method: The version of stochastic depth to use. ``"block"``
            randomly drops blocks during training. ``"sample"`` randomly drops
            samples within a block during training.
        target_layer_name: Block to replace with a stochastic block
            equivalent. The name must be registered in ``STOCHASTIC_LAYER_MAPPING``
            dictionary with the target layer class and the stochastic layer class.
            Currently, only ``'ResNetBottleneck'`` is supported.
        drop_rate: The base probability of dropping a layer or sample. Must be
            between 0.0 and 1.0.
        drop_distribution: How ``drop_rate`` is distributed across
            layers. Value must be one of ``"uniform"`` or ``"linear"``.
            ``"uniform"`` assigns the same ``drop_rate`` across all layers.
            ``"linear"`` linearly increases the drop rate across layer depth
            starting with 0 drop rate and ending with ``drop_rate``.
        drop_warmup: Percentage of training epochs to linearly
            increase the drop probability to `linear_drop_rate`. Must be between
            0.0 and 1.0.
        use_same_gpu_seed: Set to ``True`` to have the same layers dropped
            across GPUs when using multi-GPU training. Set to ``False`` to
            have each GPU drop a different set of layers. Only used
            with ``"block"`` stochastic method.
    """

    def __init__(self,
                 stochastic_method: str,
                 target_layer_name: str,
                 drop_rate: float = 0.2,
                 drop_distribution: str = 'linear',
                 drop_warmup: float = 0.0,
                 use_same_gpu_seed: bool = True):

        if drop_rate == 0.0:
            log.warning('Stochastic Depth will have no effect when drop_rate set to 0')

        if stochastic_method == "sample" and not use_same_gpu_seed:
            log.warning('use_same_gpu_seed=false has no effect when using the "sample" method')

        self.hparams = StochasticDepthHparams(stochastic_method=stochastic_method,
                                              target_layer_name=target_layer_name,
                                              drop_rate=drop_rate,
                                              drop_distribution=drop_distribution,
                                              drop_warmup=drop_warmup,
                                              use_same_gpu_seed=use_same_gpu_seed)

    @property
    def find_unused_parameters(self) -> bool:
        """DDP parameter to notify that parameters may not have gradients if it is dropped during the forward pass."""

        return (self.hparams.stochastic_method == "block")

    def match(self, event: Event, state: State) -> bool:
        """Apply on Event.INIT and Event.BATCH_START if drop_warmup is > 0.0

        Args:
            event (:class:`Event`): The current event.
            state (:class:`State`): The current state.
        Returns:
            bool: True if this algorithm should run now.        
        """

        return (event == Event.INIT) or (event == Event.BATCH_START and self.hparams.drop_warmup > 0.0)

    def apply(self, event: Event, state: State, logger: Logger) -> Optional[int]:
        """Applies StochasticDepth modification to the state's model 

        Args:
            event (Event): the current event
            state (State): the current trainer state
            logger (Logger): the training logger
        """
        assert state.model is not None
        target_layer, stochastic_layer = STOCHASTIC_LAYER_MAPPING[self.hparams.stochastic_method][
            self.hparams.target_layer_name]

        if event == Event.INIT:
            if surgery.count_module_instances(state.model, target_layer) == 0:
                log.warning(f'No {self.hparams.target_layer_name} found in model! Algorithm will function as a no-op.')

            apply_stochastic_depth(state.model,
                                   stochastic_method=self.hparams.stochastic_method,
                                   target_layer_name=self.hparams.target_layer_name,
                                   drop_rate=self.hparams.drop_rate,
                                   drop_distribution=self.hparams.drop_distribution,
                                   use_same_gpu_seed=self.hparams.use_same_gpu_seed)
            num_stochastic_layers = surgery.count_module_instances(state.model, stochastic_layer)
            logger.metric_epoch({'stochastic_depth/num_stochastic_layers': num_stochastic_layers})

        elif event == Event.BATCH_START:
            drop_warmup_iters = state.steps_per_epoch * state.max_epochs * self.hparams.drop_warmup
            if state.step < drop_warmup_iters:
                current_drop_rate = (state.step / drop_warmup_iters) * self.hparams.drop_rate
                _update_drop_rate(state.model, stochastic_layer, current_drop_rate, self.hparams.drop_distribution)
            else:
                current_drop_rate = self.hparams.drop_rate
            logger.metric_batch({'stochastic_depth/drop_rate': current_drop_rate})
