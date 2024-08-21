# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Low Precision LayerNorm."""

from __future__ import annotations

import logging
import warnings
from typing import Optional, Sequence, Union

import torch
import torch.nn.functional as F
from torch.optim import Optimizer

from composer.algorithms.warnings import NoEffectWarning
from composer.core import Algorithm, Event, Precision, State
from composer.loggers import Logger
from composer.utils import module_surgery

log = logging.getLogger(__name__)


def apply_low_precision_layernorm(
    model,
    precision: Optional[Precision] = None,
    optimizers: Optional[Union[Optimizer, Sequence[Optimizer]]] = None,
):
    if (precision != Precision.AMP_FP16 and precision != Precision.AMP_BF16):
        warnings.warn(NoEffectWarning('Low Precision LayerNorm only applies to AMP_FP16 and AMP_BF16 precisions.'))
        return model

    policy: dict[type[torch.nn.Module], module_surgery.ReplacementFunction] = {torch.nn.LayerNorm: _to_LPLayerNorm}

    replaced_instances = module_surgery.replace_module_classes(module=model, optimizers=optimizers, policies=policy)
    if len(replaced_instances) == 0:
        warnings.warn(NoEffectWarning('No instances of torch.nn.LayerNorm found.'))
    log.info(f'Successfully replaced {len(replaced_instances)} instances of LayerNorm with LowPrecisionLayerNorm')


class LowPrecisionLayerNorm(Algorithm):
    """
    Replaces all instances of :class:`torch.nn.LayerNorm` with class:`.LPLayerNorm`.

    LPLayerNorm is a thin wrapper around :class:`torch.nn.LayerNorm` which forces the layer to run
    in lower precision (torch.float16 or torch.bfloat16) if autocast is enabled. This algorithm has
    no effect in FP32 or DeepSpeed FP16 mode, where autocast is disabled.

    This algorithm is intended to be used instead of Fused LayerNorm. They have similar behavior and performance.

    Args:
        apply_at (Event): Event where algorithm is applied.
    """

    def __init__(self, apply_at: Event = Event.INIT):
        self.apply_at = apply_at
        if self.apply_at not in {Event.INIT, Event.BEFORE_LOAD, Event.AFTER_LOAD}:
            raise ValueError(
                'LowPrecisionLayerNorm only supports application on Event.INIT, Event.BEFORE_LOAD, and Event.AFTER_LOAD.',
            )

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
        apply_low_precision_layernorm(model=state.model, optimizers=state.optimizers, precision=state._precision)


class LPLayerNorm(torch.nn.LayerNorm):

    def __init__(self, normalized_shape, eps=1e-05, elementwise_affine=True, device=None, dtype=None):
        super().__init__(
            normalized_shape=normalized_shape,
            eps=eps,
            elementwise_affine=elementwise_affine,
            device=device,
            dtype=dtype,
        )

    def forward(self, x):
        module_device = x.device
        downcast_x = _cast_if_autocast_enabled(x)
        downcast_weight = _cast_if_autocast_enabled(
            self.weight,
        ) if self.weight is not None else self.weight  # pyright: ignore[reportUnnecessaryComparison]
        downcast_bias = _cast_if_autocast_enabled(
            self.bias,
        ) if self.bias is not None else self.bias  # pyright: ignore[reportUnnecessaryComparison]
        with torch.autocast(enabled=False, device_type=module_device.type):
            return F.layer_norm(downcast_x, self.normalized_shape, downcast_weight, downcast_bias, self.eps)


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


def _to_LPLayerNorm(layer: torch.nn.Module, module_index: int) -> LPLayerNorm:
    """Defines a replacement policy from a `torch.nn.LayerNorm` to a `LPLayerNorm`"""
    if not isinstance(layer, torch.nn.LayerNorm):
        raise TypeError(f'Expected torch.nn.LayerNorm, got {type(layer)}')
    lp_layernorm = LPLayerNorm(layer.normalized_shape, layer.eps, layer.elementwise_affine)

    with torch.no_grad():
        if layer.weight is None:  # pyright: ignore[reportUnnecessaryComparison]
            lp_layernorm.register_parameter('weight', None)
        else:
            lp_layernorm.weight.copy_(layer.weight)  # type: ignore
        if layer.bias is None:  # pyright: ignore[reportUnnecessaryComparison]
            lp_layernorm.register_parameter('bias', None)
        else:
            lp_layernorm.bias.copy_(layer.bias)  # type: ignore

    return lp_layernorm
