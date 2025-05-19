# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Entrypoint for distributed training (using FSDP2)."""

import logging
import time
from contextlib import contextmanager, nullcontext
from typing import Callable, Optional

import torch
from torch.distributed.fsdp.wrap import CustomPolicy

from composer.distributed.activation_checkpointing import apply_ac, generate_composer_model_check_fn
from composer.distributed.fsdp2 import prepare_fully_shard
from composer.distributed.fsdp2_utils import generate_composer_model_policy, sync_optimizer_and_model_params
from composer.distributed.param_init import meta_init
from composer.models import ComposerModel
from composer.utils.parallelism import FSDP2Config

log = logging.getLogger(__name__)


# TODO put this func into a general util function file
@contextmanager
def log_execution_time(logger: logging.Logger, operation_name: str):
    """Log the execution time of a block of code."""
    start_time = time.time()
    try:
        yield
    finally:
        end_time = time.time()
        logger.info(f'{operation_name} took {end_time - start_time:.2f} seconds')


def parallelize_model(
    model: torch.nn.Module,
    config: FSDP2Config,
    optimizer: Optional[torch.optim.Optimizer] = None,
    fsdp_wrap_policy: Optional[CustomPolicy] = None,
    activation_checkpointing_check_fn: Optional[Callable] = None,
    param_init_fn: Callable[[torch.nn.Module], None] = lambda m: None,
):
    """Prepare a model for distributed training.

    This function currently applies FSDP2 to the model, initializes parameters,
    and optionally applies activation checkpointing. It handles parameter consistency between the optimizer
    and model when an optimizer is provided.

    Args:
        model (torch.nn.Module): The model to prepare for distributed training.
        config (FSDP2Config): The configuration for FSDP distributed training.
        optimizer (Optional[torch.optim.Optimizer]): The optimizer to synchronize with the model.
            If provided, parameter states will be properly synchronized during sharding.
        fsdp_wrap_policy (Optional[CustomPolicy]): Custom policy to determine which modules should
            be wrapped with FSDP. If None, default wrapping behavior is used.
        activation_checkpointing_check_fn (Optional[Callable]): Function that determines whether a
            module's activations should be checkpointed or offloaded. Only used when activation
            checkpointing or CPU offloading is enabled in the config.
        param_init_fn (Callable[[torch.nn.Module], None]): Function to initialize model parameters
            after FSDP wrapping. Defaults to a no-op function.

    Raises:
        ValueError: If the config is not an FSDP2Config or if activation_checkpointing_check_fn is provided
            but neither activation_checkpointing nor activation_cpu_offload is enabled in the config.
    """
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
        with log_execution_time(log, 'Prepare FSDP2'):
            prepare_fully_shard(model, config, fsdp_wrap_policy)
        with log_execution_time(log, 'Meta Init Device'):
            param_init_fn(model)
        # NOTE appy_ac can not be included in this context as it would wrap and replace the sub-modules thus disqualify FQN of params


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
    activation_checkpointing_check_fn = generate_composer_model_check_fn(
        composer_model,
    ) if config.activation_checkpointing or config.activation_cpu_offload else None
    parallelize_model(
        composer_model,
        config,
        optimizer=optimizer,
        fsdp_wrap_policy=generate_composer_model_policy(composer_model),
        activation_checkpointing_check_fn=activation_checkpointing_check_fn,
        param_init_fn=meta_init,
    )
