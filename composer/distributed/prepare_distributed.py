# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Entrypoint for distributed training (using FSDP2)."""

import logging
import time
from contextlib import contextmanager, nullcontext
from typing import Callable, Optional, Union

import torch
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
)
from torch.distributed.fsdp.wrap import CustomPolicy

from composer.core.precision import Precision, _validate_precision
from composer.devices import DeviceGPU
from composer.distributed.activation_checkpointing import apply_ac, generate_composer_model_check_fn
from composer.distributed.fsdp2 import prepare_fully_shard, sync_module_states
from composer.distributed.fsdp2_utils import generate_composer_model_policy, sync_optimizer_and_model_params
from composer.distributed.param_init import meta_init
from composer.distributed.shared_utils import update_sync_module_states_if_needed
from composer.models import ComposerModel
from composer.utils import dist
from composer.utils.device import get_device
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


@contextmanager
def get_full_state_dict(model: torch.nn.Module):
    """Context manager to temporarily get full state dict regardless of should_save_peft_only setting for huggingface models.

    PEFT models with lora have an updated state_dict fn (in composer/models/huggingface.py) that
    returns the state_dict with only the lora params if should_save_peft_only is True.
    But when we're syncing module states, we need the full state dict, so we temporarily set
    should_save_peft_only to False.
    """
    # TODO: Since sharding peft/lora weights can be inefficient due to their small sizes (leading to communication overhead
    # outweighing memory savings), we should provide an interface that allows users to avoid sharding these weights.
    original_setting = getattr(model, 'should_save_peft_only', None)
    if original_setting is not None:
        model.should_save_peft_only = False  # type: ignore
    try:
        yield
    finally:
        if original_setting is not None:
            model.should_save_peft_only = original_setting  # type: ignore


def _check_duplicate_modules(model: torch.nn.Module) -> None:
    """Checks whether the model has duplicate module references.

    This detects cases where the same module object is referenced multiple times
    in the model hierarchy (e.g., self.net = Linear(...); self.net2 = self.net).
    This is different from weight tying, where different modules share parameters.

    This is a legalization for the fact that FSDP2 does not support duplicate module
    references in the model hierarchy for mixed init and/or monolithic checkpointing.

    If you do not have this legalization, you will encounter errors like:
    "...got mixed torch.Tensor and DTensor..." in the sync_module_states step.
    """
    all_modules = set(dict(model.named_modules(remove_duplicate=False)).keys())
    deduplicated_modules = set(dict(model.named_modules(remove_duplicate=True)).keys())

    duplicate_modules = all_modules - deduplicated_modules
    if duplicate_modules:
        raise ValueError(
            f'Model has duplicate module references. Modules {duplicate_modules} '
            f'are the same object as previously encountered modules. '
            f'This is not supported by FSDP2. Please ensure each module reference '
            f'is unique (weight tying through parameter sharing is still allowed).',
        )


def _parallelize_model_helper(
    model: torch.nn.Module,
    config: FSDP2Config,
    precision: Precision,
    fsdp_wrap_policy: Optional[CustomPolicy] = None,
    param_init_fn: Callable[[torch.nn.Module], None] = lambda m: None,
) -> None:
    """Helper function for parallelizing a model.

    There are two paths with how we parallelize a model:
    1. Without sync_module_states: fully_shard first, then param_init. This prevents OOM by avoiding
       redundant parameter copies across all ranks during param_init.
    2. With sync_module_states: param_init on rank 0 first, then fully_shard, then broadcast the
       initialized state to all other ranks. This makes sure that all ranks have rank 0's model state.
    """
    update_sync_module_states_if_needed(model, config)

    if config.sync_module_states:
        # Check for duplicate module references which will cause issues with sync_module_states
        _check_duplicate_modules(model)

        # If we are syncing module states, we assume that rank 0 has the model on CPU/GPU
        # and the params are already initialized on rank 0.
        full_state_dict = {}
        if dist.get_global_rank() == 0:
            # Get the full state dict of the model offloaded to CPU
            options = StateDictOptions(
                full_state_dict=True,
                cpu_offload=True,
            )
            with get_full_state_dict(model):
                full_state_dict = get_model_state_dict(model, options=options)

        with log_execution_time(log, 'Prepare FSDP2'):
            prepare_fully_shard(model, config, precision, fsdp_wrap_policy)

        with log_execution_time(log, 'Sync Module States across all ranks'):
            sync_module_states(model, full_state_dict)
    else:
        with log_execution_time(log, 'Prepare FSDP2'):
            prepare_fully_shard(model, config, precision, fsdp_wrap_policy)

        with log_execution_time(log, 'Parameter Initialization'):
            param_init_fn(model)


def parallelize_model(
    model: torch.nn.Module,
    config: FSDP2Config,
    optimizer: Optional[torch.optim.Optimizer] = None,
    precision: Optional[Union[str, Precision]] = None,
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
        precision (Precision): The precision to use for the model. Defaults to AMP_FP16 for GPU and FP32 for CPU.
            It doesn't have an optional type because `parallelize_composer_model` already sets the precision.
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
    if precision is None:
        precision = Precision.AMP_FP16 if isinstance(get_device(), DeviceGPU) else Precision.FP32
    elif isinstance(precision, str):
        precision = Precision(precision)
    _validate_precision(precision, get_device())

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
        _parallelize_model_helper(model, config, precision, fsdp_wrap_policy, param_init_fn)
        # NOTE appy_ac can not be included in this context as it would wrap and replace the sub-modules thus disqualify FQN of params


def parallelize_composer_model(
    composer_model: ComposerModel,
    optimizer: Optional[torch.optim.Optimizer],
    config: FSDP2Config,
    precision: Optional[Precision] = None,
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
        precision (Precision): The precision to use for the model. Defaults to AMP_FP16 for GPU and FP32 for CPU.
    """
    assert isinstance(composer_model, ComposerModel), f'{type(composer_model)} is not a ComposerModel'
    activation_checkpointing_check_fn = generate_composer_model_check_fn(
        composer_model,
    ) if config.activation_checkpointing or config.activation_cpu_offload else None
    parallelize_model(
        composer_model,
        config,
        optimizer=optimizer,
        precision=precision,
        fsdp_wrap_policy=generate_composer_model_policy(composer_model),
        activation_checkpointing_check_fn=activation_checkpointing_check_fn,
        param_init_fn=meta_init,
    )
