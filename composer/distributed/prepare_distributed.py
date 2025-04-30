# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Entrypoint for distributed training (using FSDP2)."""

from typing import Callable, Optional

import torch
from torch.distributed.fsdp.wrap import CustomPolicy

from composer.distributed.activation_checkpointing import apply_ac
from composer.distributed.fsdp2 import prepare_fully_shard
from composer.utils.parallelism import FSDP2Config, FSDPConfig


def parallelize_model(
    model: torch.nn.Module,
    config: FSDP2Config | FSDPConfig,
    optimizer: Optional[torch.optim.Optimizer] = None,
    auto_wrap_policy: Optional[CustomPolicy] = None,
    activation_checkpointing_check_fn: Optional[Callable] = None,
):
    """Prepare a model for distributed training.

    Args:
        model (torch.nn.Module): The model to prepare for distributed training.
        config (FSDP2Config | FSDPConfig): The configuration for distributed training. Currently only FSDP2Config is supported.
        optimizer (Optional[torch.optim.Optimizer]): The optimizer to use for distributed training.
        auto_wrap_policy (Optional[CustomPolicy]): The auto wrap policy to use for distributed training.
        activation_checkpointing_check_fn (Optional[Callable]): The function to use to check if a module's activations should be checkpointed or offloaded.
    """
    if isinstance(config, FSDPConfig):
        raise ValueError('FSDPConfig is not supported for now, use FSDP2Config instead')

    if activation_checkpointing_check_fn is not None:
        assert config.activation_checkpointing or config.activation_cpu_offload, 'Activation checkpointing or offloading must be enabled if activation_checkpointing_check_fn is provided'

    prepare_fully_shard(model, optimizer, config, auto_wrap_policy)

    if config.activation_checkpointing or config.activation_cpu_offload:
        apply_ac(model, config, activation_checkpointing_check_fn)
