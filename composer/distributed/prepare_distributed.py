# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Entrypoint for distributed training (using FSDP2)."""

from typing import Callable, Optional
from contextlib import nullcontext

import torch
from torch.distributed.fsdp.wrap import CustomPolicy

from composer.models import ComposerModel
from composer.distributed.activation_checkpointing import apply_ac
from composer.distributed.fsdp2 import prepare_fully_shard
from composer.utils.parallelism import FSDP2Config
from composer.distributed.fsdp2_utils import sync_optimizer_and_model_params
from composer.distributed.param_init import meta_init
from composer.distributed.fsdp2_utils import generate_fsdp1_composer_model_policy
from composer.distributed.activation_checkpointing import generate_fsdp1_composer_check_fn


def parallelize_model(
    model: torch.nn.Module,
    config: FSDP2Config,
    optimizer: Optional[torch.optim.Optimizer] = None,
    fsdp_wrap_policy: Optional[CustomPolicy] = None,
    activation_checkpointing_check_fn: Optional[Callable] = None,
    param_init_fn: Callable[[torch.nn.Module], None] = lambda m: None,
):
    """Prepare a model for distributed training.

    Args:
        model (torch.nn.Module): The model to prepare for distributed training.
        config FSDP2Config: The configuration for distributed training.
        optimizer (Optional[torch.optim.Optimizer]): The optimizer to use for distributed training.
        fsdp_wrap_policy (Optional[CustomPolicy]): The FSDP wrap policy to use for distributed training.
        activation_checkpointing_check_fn (Optional[Callable]): The function to use to check if a module's activations should be checkpointed or offloaded.
        param_init_fn (Callable[[torch.nn.Module], None]): The function to use to initialize the model's parameters.
    """
    if not isinstance(config, FSDP2Config):
        raise ValueError('FSDP2Config is the only supported config for now')

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
    # Use the context manager for optimizer synchronization if optimizer is provided
    with sync_optimizer_and_model_params(optimizer, model) if optimizer is not None else nullcontext():
        prepare_fully_shard(model, config, fsdp_wrap_policy)
        # print(model)
        # for name, param in model.named_parameters():
        #     print(name)
        param_init_fn(model)

def parallelize_composer_model(
    composer_model: ComposerModel,
    optimizer: Optional[torch.optim.Optimizer],
    config: FSDP2Config,
):
    """Prepare a ComposerModel for distributed training.

    NOTE we apply parallelization to each of the composer model's submodules to provide compatibility with models defined for FSDP1.
    This is not strictly necessary for FSDP2 as it relies on DTensor so even if a module is not wrapped with FSDP2 and its params are sharded,
    it is still functional (but potentially less performant due to lack of grouped prefetching etc).

    For advanced users who want to have access to more flexible fsdp_wrap_policy or activation_checkpointing_check_fn, they should use `parallelize_model` directly.
    
    Args:
        composer_model (ComposerModel): The ComposerModel to prepare for distributed training.
        optimizer (Optional[torch.optim.Optimizer]): The optimizer to use for distributed training.
        config (FSDP2Config): The configuration for distributed training. Currently only FSDP2Config is supported.
    """

    assert isinstance(composer_model, ComposerModel), f'{type(composer_model)} is not a ComposerModel'
    activation_checkpointing_check_fn = generate_fsdp1_composer_check_fn(composer_model) if config.activation_checkpointing or config.activation_cpu_offload else None
    parallelize_model(composer_model, config, optimizer=optimizer, fsdp_wrap_policy=generate_fsdp1_composer_model_policy(composer_model), activation_checkpointing_check_fn=activation_checkpointing_check_fn, param_init_fn=meta_init)
