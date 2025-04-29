# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Helpers for activation checkpointing. Note that while this is orthogonal to FSDP2, it is implemented in the distributed directory because it is closely related to FSDP2."""

from typing import Callable

import torch
import torch.nn as nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    apply_activation_checkpointing,
    checkpoint_wrapper,
    offload_wrapper,
)

from composer.utils.parallelism import FSDP2Config


def apply_ac(model: nn.Module, fsdp2_config: FSDP2Config) -> None:
    """Apply activation checkpointing to the model. This is orthogonal to FSDP2 so it can be applied pre-sharding or post-sharding.

    This method follows the same logic as FSDP1 as well as TorchTitan's AC example. FSDP1 doesn't support op-based
    selective activation checkpointing, so that will not be implemented unless we get sufficient demand.

    Args:
        model (nn.Module): The model to apply activation checkpointing to.
        fsdp2_config (FSDP2Config): The FSDP2 configuration.
    """
    activation_checkpointing = fsdp2_config.activation_checkpointing
    activation_cpu_offload = fsdp2_config.activation_cpu_offload

    # Create the base checkpointing wrapper using no_reentrant checkpointing by default as
    # PyTorch notes that reentrant checkpointing is deprecated and will be removed in a future release
    opt_checkpoint_wrapper = lambda m: checkpoint_wrapper(
        m,
        checkpoint_impl=CheckpointImpl.NO_REENTRANT,
    ) if activation_checkpointing else (lambda module: module)
    # Create the combined wrapper which takes cpu offloading into consideration
    opt_combined_wrapper = (
        lambda module: offload_wrapper(
            opt_checkpoint_wrapper(module)
            if activation_checkpointing else module,  # type: ignore reportGeneralTypeIssues
        )
    ) if activation_cpu_offload else opt_checkpoint_wrapper

    # Create the check function to determine if a module should be checkpointed
    def _check_fn(module: torch.nn.Module) -> bool:
        if hasattr(module, '_activation_checkpointing'):
            return bool(module._activation_checkpointing)
        if hasattr(
            model,
            'activation_checkpointing_fn',
        ) and isinstance(model.activation_checkpointing_fn, Callable):
            return model.activation_checkpointing_fn(module)
        return False

    # Apply the activation checkpointing on the model, this uses _recursive_wrap to apply the wrapper all submodules
    # but doesn't apply the wrapper to the root module
    apply_activation_checkpointing(model, opt_combined_wrapper, _check_fn)  # type: ignore
