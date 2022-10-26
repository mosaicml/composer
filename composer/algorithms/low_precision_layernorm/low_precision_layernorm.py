# Copyright 2022 MosaicML Agent authors
# SPDX-License-Identifier: Apache-2.0

# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

# Copyright 2022 MosaicML. All Rights Reserved.

from __future__ import annotations

import logging
import warnings
from typing import Dict, Optional, Sequence, Type, Union

import torch
import torch.nn.functional as F

from composer.algorithms.warnings import NoEffectWarning
from composer.core import Algorithm, Event, State
from composer.loggers import Logger
from composer.utils import module_surgery

log = logging.getLogger(__name__)


def _cast_if_autocast_enabled(hidden_states):
    if not torch.is_autocast_enabled():
        return hidden_states
    else:
        return torch.cuda.amp.autocast_mode._cast(hidden_states, torch.get_autocast_gpu_dtype())


class LPLayerNorm(torch.nn.LayerNorm):

    def __init__(self, layer):
        super().__init__(normalized_shape=layer.normalized_shape,
                         eps=layer.eps,
                         elementwise_affine=layer.elementwise_affine)

        with torch.no_grad():
            self.weight.copy_(layer.weight)
            self.bias.copy_(layer.bias)

    def forward(self, x):
        module_device = x.device
        downcast_x = _cast_if_autocast_enabled(x)
        downcast_weight = _cast_if_autocast_enabled(self.weight)
        downcast_bias = _cast_if_autocast_enabled(self.bias)
        with torch.autocast(enabled=False, device_type=module_device.type):
            return F.layer_norm(downcast_x, self.normalized_shape, downcast_weight, downcast_bias, self.eps)


def from_Layer(layer: torch.nn.Module, module_index: int) -> LPLayerNorm:
    assert isinstance(layer,
                      torch.nn.LayerNorm), 'The replacement policy will look for all instances of torch.nn.LayerNorm'
    return LPLayerNorm(layer)


def apply_low_precision_layernorm(model, optimizers: Union[torch.optim.Optimizer, Sequence[torch.optim.Optimizer]]):

    policy: Dict[Type[torch.nn.Module], module_surgery.ReplacementFunction] = {torch.nn.LayerNorm: from_Layer}

    replaced_instances = module_surgery.replace_module_classes(module=model, optimizers=optimizers, policies=policy)
    if len(replaced_instances) == 0:
        warnings.warn(NoEffectWarning('No instances of torch.nn.LayerNorm found.'))
    log.info(f'Successfully replaced {len(replaced_instances)} instances of LayerNorm')

    return model


class LowPrecisionLayerNorm(Algorithm):
    """
    Replaces all instances of `torch.nn.LayerNorm` with `composer.algorithms.low_precision_layernorm.low_precision_layernorm.LPLayerNorm`.

    LPLayerNorm is a thin wrapper around `torch.nn.LayerNorm` which forces the layer to run in lower precision (torch.float16 or torch.bfloat16)
    if autocast is enabled.

    This algorithm is intended to be used instead of Fused LayerNorm. They have similar behavior and performance.

    Args:
        apply_at (Event, optional): Event where algorithm is applied.
    """

    def __init__(self, apply_at: Optional[Event] = None):
        self.apply_at = Event.INIT if apply_at is None else apply_at
        if self.apply_at not in {Event.INIT, Event.AFTER_LOAD}:
            raise ValueError('LowPrecisionLayerNorm only supports application on Event.INIT and Event.AFTER_LOAD.')

    def match(self, event: Event, state: State) -> bool:
        del state  # unused
        return event == self.apply_at

    def apply(self, event: Event, state: State, logger: Logger) -> Optional[int]:
        del event, logger  # unused
        apply_low_precision_layernorm(model=state.model, optimizers=state.optimizers)
