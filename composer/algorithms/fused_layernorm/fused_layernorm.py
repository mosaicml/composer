# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

# Copyright 2022 MosaicML. All Rights Reserved.

from __future__ import annotations

import logging
from typing import Optional, Sequence, Union

import torch

try:
    from apex.normalization.fused_layer_norm import FusedLayerNorm as APEXFusedLayerNorm
except ImportError as e:
    raise ImportError(
        "https://github.com/NVIDIA/apex is not installed, and therefore the Fused LayerNorm algorithm cannot be applied."
    ) from e

from composer.core import Algorithm, Event, State
from composer.loggers import Logger
from composer.utils import module_surgery

log = logging.getLogger(__name__)


def apply_fused_layernorm(model: torch.nn.Module, optimizers: Union[torch.optim.Optimizer,
                                                                    Sequence[torch.optim.Optimizer]]) -> None:
    """
    Replaces all instances of `torch.nn.LayerNorm` with a `apex.normalization.fused_layer_norm.FusedLayerNorm <https://nvidia.github.io/apex/layernorm.html>`_.

    By fusing multiple kernel launches into one, this usually improves GPU utilization.

    """

    # get new parameter values
    d_embeds = []
    layernorm_eps = []
    for module_name, module_class in model.named_modules():
        if isinstance(module_class, torch.nn.LayerNorm):
            assert len(module_class.normalized_shape) == 1
            d_embeds.append(module_class.normalized_shape[0])
            layernorm_eps.append(module_class.eps)

    # ensure that the model contains the same d_embed and layernorm.eps throughout
    for l in [d_embeds, layernorm_eps]:
        assert len(set(l)) == 1

    d_embed = d_embeds[0]
    layernorm_eps = layernorm_eps[0]

    # prepare the replacement policy and perform replacement
    policy = {torch.nn.LayerNorm: lambda x, module_index: APEXFusedLayerNorm(normalized_shape=d_embed, eps=layernorm_eps)}
    replaced_instances = module_surgery.replace_module_classes(module=model, optimizers=optimizers, policies=policy)
    log.info(f"Successfully replaced {len(replaced_instances)} of LayerNorm with a Fused LayerNorm.")


class FusedLayerNorm(Algorithm):
    """Replaces all instances of `torch.nn.LayerNorm` with a `apex.normalization.fused_layer_norm.FusedLayerNorm <https://nvidia.github.io/apex/layernorm.html>`_.

    By fusing multiple kernel launches into one, this usually improves GPU utilization.

    Runs on ``Event.INIT``, so it can replace all instances of `torch.nn.LayerNorm` before the model is DDP wrapped. Has no hyperparameters.

    Example:
        .. testcode::

            from composer.algorithms import FusedLayerNorm
            algorithm = FusedLayerNorm()
            trainer = Trainer(
                model=model,
                train_dataloader=train_dataloader,
                eval_dataloader=eval_dataloader,
                max_duration="1ep",
                algorithms=[algorithm],
                optimizers=[optimizer]
            )
    """

    def __init__(self):
        # FusedLayerNorm takes no arguments
        pass

    def match(self, event: Event, state: State) -> bool:
        del state  # unused
        return event == Event.INIT

    def apply(self, event: Event, state: State, logger: Logger) -> Optional[int]:
        del event, logger  # unused
        apply_fused_layernorm(model=state.model, optimizers=state.optimizers)
