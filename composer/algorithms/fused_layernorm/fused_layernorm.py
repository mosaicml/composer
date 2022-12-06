# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

# Copyright 2022 MosaicML. All Rights Reserved.

from __future__ import annotations

import logging
import warnings
from typing import Dict, Optional, Sequence, Type, Union

import torch

try:
    from apex.normalization.fused_layer_norm import FusedLayerNorm as APEXFusedLayerNorm
    APEX_INSTALLED = True
except ImportError as e:
    APEX_INSTALLED = False

from composer.algorithms.warnings import NoEffectWarning
from composer.core import Algorithm, Event, State
from composer.loggers import Logger
from composer.utils import module_surgery

log = logging.getLogger(__name__)


def check_if_apex_installed():
    if not APEX_INSTALLED:
        raise ImportError(
            'https://github.com/NVIDIA/apex is not installed. The Fused LayerNorm algorithm cannot be applied. The MosaicML Docker Images (https://hub.docker.com/r/mosaicml/pytorch) contain a copy of APEX for easy use.'
        )


def warn_fln_deprecation():
    warnings.warn(
        DeprecationWarning(
            'Fused LayerNorm will be deprecated and removed in v0.13. Please use Low Precision LayerNorm instead.'))


def from_LayerNorm(layer: torch.nn.Module, module_index: int) -> APEXFusedLayerNorm:
    """Defines a replacement policy from a `torch.nn.LayerNorm` to a `apex.normalization.fused_layer_norm`"""
    assert isinstance(layer,
                      torch.nn.LayerNorm), 'The replacement policy will look for all instances of torch.nn.LayerNorm'
    return APEXFusedLayerNorm(normalized_shape=layer.normalized_shape, eps=layer.eps)


def apply_fused_layernorm(model: torch.nn.Module, optimizers: Union[torch.optim.Optimizer,
                                                                    Sequence[torch.optim.Optimizer]]) -> None:
    """Replaces all instances of `torch.nn.LayerNorm` with a `apex.normalization.fused_layer_norm.FusedLayerNorm
    <https://nvidia.github.io/apex/layernorm.html>`_.

    By fusing multiple kernel launches into one, this usually improves GPU utilization.
    """
    check_if_apex_installed()

    # prepare the replacement policy and perform replacement
    policy: Dict[Type[torch.nn.Module], module_surgery.ReplacementFunction] = {torch.nn.LayerNorm: from_LayerNorm}
    replaced_instances = module_surgery.replace_module_classes(module=model, optimizers=optimizers, policies=policy)
    if len(replaced_instances) == 0:
        warnings.warn(
            NoEffectWarning(
                'No instances of `torch.nn.LayerNorm` were found, and therefore, there were no modules to replace.'))
    log.info(f'Successfully replaced {len(replaced_instances)} of LayerNorm with a Fused LayerNorm.')


class FusedLayerNorm(Algorithm):
    """Replaces all instances of `torch.nn.LayerNorm` with a `apex.normalization.fused_layer_norm.FusedLayerNorm
    <https://nvidia.github.io/apex/layernorm.html>`_.

    By fusing multiple kernel launches into one, this usually improves GPU utilization.

    Runs on ``Event.INIT``, so it can replace all instances of `torch.nn.LayerNorm` before the model is DDP wrapped. Has no hyperparameters.

    Example:
        .. testsetup::

           def no_op(self, *args): pass

           from composer.algorithms import FusedLayerNorm

           FusedLayerNorm.__init__ = no_op

           FusedLayerNorm.apply = no_op

           model, train_dataloader, optimizer = _make_synthetic_bert_state()

        .. testcode::

           from composer.algorithms import FusedLayerNorm

           algorithm = FusedLayerNorm()
           trainer = Trainer(
               model=model,
               train_dataloader=train_dataloader,
               max_duration="1ep",
               algorithms=[algorithm],
               optimizers=[optimizer]
           )
    """

    def __init__(self):
        # FusedLayerNorm takes no arguments
        warn_fln_deprecation()
        check_if_apex_installed()

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'

    @staticmethod
    def required_on_load() -> bool:
        return True

    def match(self, event: Event, state: State) -> bool:
        del state  # unused
        return event == Event.INIT

    def apply(self, event: Event, state: State, logger: Logger) -> Optional[int]:
        del event, logger  # unused
        apply_fused_layernorm(model=state.model, optimizers=state.optimizers)
