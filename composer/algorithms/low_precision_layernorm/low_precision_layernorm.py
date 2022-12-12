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
from packaging import version

from composer.algorithms.warnings import NoEffectWarning
from composer.core import Algorithm, Event, Precision, State
from composer.loggers import Logger
from composer.utils import module_surgery

log = logging.getLogger(__name__)

try:
    from apex.normalization.fused_layer_norm import FusedLayerNorm as APEXFusedLayerNorm
    APEX_INSTALLED = True
except ImportError as e:
    APEX_INSTALLED = False


def _cast_if_autocast_enabled(hidden_states):
    if not torch.is_autocast_enabled():
        return hidden_states
    else:
        return torch.cuda.amp.autocast_mode._cast(hidden_states, torch.get_autocast_gpu_dtype())


def check_if_apex_installed():
    if not APEX_INSTALLED:
        raise ImportError(
            'https://github.com/NVIDIA/apex is not installed. The Low Precision LayerNorm algorithm cannot be applied. The MosaicML Docker Images (https://hub.docker.com/r/mosaicml/pytorch) contain a copy of APEX for easy use.'
        )


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


def to_LPLayerNorm(layer: torch.nn.Module, module_index: int) -> LPLayerNorm:
    assert isinstance(layer,
                      torch.nn.LayerNorm), 'The replacement policy will look for all instances of torch.nn.LayerNorm'
    return LPLayerNorm(layer)


def to_FusedLayerNorm(layer: torch.nn.Module, module_index: int) -> APEXFusedLayerNorm:
    """Defines a replacement policy from a `torch.nn.LayerNorm` to a `apex.normalization.fused_layer_norm`"""
    assert isinstance(layer,
                      torch.nn.LayerNorm), 'The replacement policy will look for all instances of torch.nn.LayerNorm'
    fused_layernorm = APEXFusedLayerNorm(normalized_shape=layer.normalized_shape, eps=layer.eps)
    with torch.no_grad():
        fused_layernorm.weight.copy_(layer.weight)
        fused_layernorm.bias.copy_(layer.bias)
    return fused_layernorm


def apply_low_precision_layernorm(model, optimizers: Union[torch.optim.Optimizer, Sequence[torch.optim.Optimizer]],
                                  precision: Precision):

    if (precision != Precision.AMP_FP16 and precision != Precision.AMP_BF16):
        warnings.warn(NoEffectWarning('Low Precision LayerNorm only applies to AMP_FP16 and AMP_BF16 precisions.'))
        return model

    policy: Dict[Type[torch.nn.Module], module_surgery.ReplacementFunction] = {torch.nn.LayerNorm: to_LPLayerNorm}

    # Prior to v1.13, torch.nn.LayerNorm is slow in bf16 precision.
    # We use FusedLayerNorm as a fallback.
    if version.parse(torch.__version__) < version.parse('1.13') and precision == Precision.AMP_BF16:
        check_if_apex_installed()
        policy: Dict[Type[torch.nn.Module], module_surgery.ReplacementFunction] = {
            torch.nn.LayerNorm: to_FusedLayerNorm
        }

    replaced_instances = module_surgery.replace_module_classes(module=model, optimizers=optimizers, policies=policy)
    if len(replaced_instances) == 0:
        warnings.warn(NoEffectWarning('No instances of torch.nn.LayerNorm found.'))
    log.info(f'Successfully replaced {len(replaced_instances)} instances of LayerNorm with LowPrecisionLayerNorm')

    return model


class LowPrecisionLayerNorm(Algorithm):
    """
    Replaces all instances of `torch.nn.LayerNorm` with `composer.algorithms.low_precision_layernorm.low_precision_layernorm.LPLayerNorm`.

    LPLayerNorm is a thin wrapper around `torch.nn.LayerNorm` which forces the layer to run in lower precision (torch.float16 or torch.bfloat16)
    if autocast is enabled. This algorithm has no effect in FP32 or DeepSpeed FP16 mode, where autocast is disabled.

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
        apply_low_precision_layernorm(model=state.model, optimizers=state.optimizers, precision=state._precision)
