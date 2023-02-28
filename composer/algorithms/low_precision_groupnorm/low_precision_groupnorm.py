# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Low Precision GroupNorm."""

from __future__ import annotations

import logging
import warnings
from typing import Dict, Optional, Sequence, Type, Union

import torch
import torch.nn.functional as F
from torch.optim import Optimizer

from composer.algorithms.warnings import NoEffectWarning
from composer.core import Algorithm, Event, Precision, State
from composer.loggers import Logger
from composer.utils import module_surgery

log = logging.getLogger(__name__)


def apply_low_precision_groupnorm(model,
                                  precision: Optional[Precision] = None,
                                  optimizers: Optional[Union[Optimizer, Sequence[Optimizer]]] = None):
    if (precision != Precision.AMP_FP16 and precision != Precision.AMP_BF16):
        warnings.warn(NoEffectWarning('Low Precision GroupNorm only applies to AMP_FP16 and AMP_BF16 precisions.'))
        return model

    policy: Dict[Type[torch.nn.Module], module_surgery.ReplacementFunction] = {torch.nn.GroupNorm: _to_LPGroupNorm}

    replaced_instances = module_surgery.replace_module_classes(module=model, optimizers=optimizers, policies=policy)
    if len(replaced_instances) == 0:
        warnings.warn(NoEffectWarning('No instances of torch.nn.GroupNorm found.'))
    log.info(f'Successfully replaced {len(replaced_instances)} instances of GroupNorm with LowPrecisionGroupNorm')


class LowPrecisionGroupNorm(Algorithm):
    """
    Replaces all instances of :class:`torch.nn.GroupNorm` with class:`.LPGroupNorm`.

    LPGroupNorm is a thin wrapper around :class:`torch.nn.GroupNorm` which forces the layer to run
    in lower precision (torch.float16 or torch.bfloat16) if autocast is enabled. This algorithm has
    no effect in FP32 or DeepSpeed FP16 mode, where autocast is disabled.

    This algorithm is intended to be used instead of Fused GroupNorm. They have similar behavior and performance.

    Args:
        apply_at (Event): Event where algorithm is applied.
    """

    def __init__(self, apply_at: Event = Event.INIT):
        self.apply_at = apply_at
        if self.apply_at not in {Event.INIT, Event.AFTER_LOAD}:
            raise ValueError('LowPrecisionGroupNorm only supports application on Event.INIT and Event.AFTER_LOAD.')

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(apply_at={self.apply_at})'

    @staticmethod
    def required_on_load() -> bool:
        return True

    def match(self, event: Event, state: State) -> bool:
        del state  # unused
        return event == self.apply_at

    def apply(self, event: Event, state: State, logger: Logger) -> Optional[int]:
        del event, logger  # unused
        apply_low_precision_groupnorm(model=state.model, optimizers=state.optimizers, precision=state._precision)


class LPGroupNorm(torch.nn.GroupNorm):

    def __init__(self, layer):
        super().__init__(
            num_groups=layer.num_groups,
            num_channels=layer.num_channels,
            eps=layer.eps,
            affine=layer.affine,
        )

        with torch.no_grad():
            self.weight.copy_(layer.weight)
            self.bias.copy_(layer.bias)

    def forward(self, x):
        module_device = x.device
        downcast_x = _cast_if_autocast_enabled(x)
        downcast_weight = _cast_if_autocast_enabled(self.weight)
        downcast_bias = _cast_if_autocast_enabled(self.bias)
        with torch.autocast(enabled=False, device_type=module_device.type):
            return F.group_norm(downcast_x, self.num_groups, downcast_weight, downcast_bias, self.eps)


def _cast_if_autocast_enabled(tensor):
    if torch.is_autocast_enabled():
        if tensor.device.type == 'cuda':
            dtype = torch.get_autocast_gpu_dtype()
        elif tensor.device.type == 'cpu':
            dtype = torch.get_autocast_cpu_dtype()
        else:
            raise NotImplementedError()
        return tensor.to(dtype=dtype)
    return tensor


def _to_LPGroupNorm(layer: torch.nn.Module, module_index: int) -> LPGroupNorm:
    if not isinstance(layer, torch.nn.GroupNorm):
        raise TypeError(f'Expected torch.nn.GroupNorm, got {type(layer)}')
    return LPGroupNorm(layer)
