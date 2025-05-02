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
    fsdp_wrap_policy: Optional[CustomPolicy] = None,
    activation_checkpointing_check_fn: Optional[Callable] = None,
    auto_microbatching: bool = False,
) -> tuple[list, dict]:
    """Prepare a model for distributed training.

    Args:
        model (torch.nn.Module): The model to prepare for distributed training.
        config (FSDP2Config | FSDPConfig): The configuration for distributed training. Currently only FSDP2Config is supported.
        optimizer (Optional[torch.optim.Optimizer]): The optimizer to use for distributed training.
        fsdp_wrap_policy (Optional[CustomPolicy]): The FSDP wrap policy to use for distributed training.
        activation_checkpointing_check_fn (Optional[Callable]): The function to use to check if a module's activations should be checkpointed or offloaded.
        auto_microbatching (bool): Whether to use auto microbatching.

    Returns:
        List[torch.utils.hooks.RemovableHandle]: A list of removable hook handles for the OOM hooks if auto_microbatching is enabled.
        Dict[str, nn.Module]: A dictionary of the named modules after fully sharding.
    """
    if isinstance(config, FSDPConfig):
        raise ValueError('FSDPConfig is not supported for now, use FSDP2Config instead')

    if activation_checkpointing_check_fn is not None:
        if not config.activation_checkpointing and not config.activation_cpu_offload:
            raise ValueError(
                'Activation checkpointing or offloading must be enabled if activation_checkpointing_check_fn is provided',
            )

    if config.activation_checkpointing or config.activation_cpu_offload:
        apply_ac(
            model,
            config.activation_checkpointing,
            config.activation_cpu_offload,
            activation_checkpointing_check_fn,
        )

    hook_handles, named_modules = prepare_fully_shard(model, optimizer, config, fsdp_wrap_policy, auto_microbatching)

    return hook_handles, named_modules
