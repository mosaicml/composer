# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Helpers for FSDP2."""

from typing import Callable, Optional

import torch
import torch.nn as nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    apply_activation_checkpointing,
    checkpoint_wrapper,
    offload_wrapper,
)
from torch.distributed.fsdp._fully_shard import fully_shard
from torch.distributed.fsdp.wrap import CustomPolicy

from composer.distributed.fsdp2_utils import (
    check_param_tying,
    generate_default_policy,
    get_standalone_and_tied_modules,
    legalize_param_sharing_between_modules,
    update_optimizer_modules,
)
from composer.utils.parallelism import FSDP2Config


def apply_ac(model: nn.Module, fsdp2_config: FSDP2Config) -> None:
    """Apply activation checkpointing to the model. This is orthogonal to FSDP2 so it can be applied pre-sharding or post-sharding.

    This method follows the same logic as FSDP1 as well as TorchTitan's AC example.

    Args:
        model (nn.Module): The model to apply activation checkpointing to.
        fsdp2_config (FSDP2Config): The FSDP2 configuration.
    """
    # TODO: Add support for selective activation checkpointing
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
    apply_activation_checkpointing(model, opt_combined_wrapper, _check_fn)


def _recursive_apply_fully_shard(
    root_module: nn.Module,
    module: nn.Module,
    target_modules_to_kwargs: dict[nn.Module, dict],
) -> None:
    """Recursive helper to apply fully_shard based on policy and legalization.

    Args:
        root_module (nn.Module): The root module to check for parameter sharing.
        module (nn.Module): The current module being processed.
        target_modules_to_kwargs (dict[nn.Module, dict]): Dictionary mapping modules to their fully_shard kwargs.

    Returns:
        None (fully_shards modules in place)
    """
    # 1. Identify direct children candidates for sharding based on whether they are in target_modules_to_kwargs
    child_candidates = [child for child in module.children() if child in target_modules_to_kwargs]

    # 2. Legalize child candidates
    standalone_child_candidates: list[nn.Module] = []
    if child_candidates:
        # Check for tying among the valid candidates based on the policy
        standalone_child_candidates, tied_children = get_standalone_and_tied_modules(child_candidates)
        if tied_children:
            raise ValueError(
                f'Detected tied parameters between modules designated for FSDP wrapping within {type(module).__name__}. '
                f'FSDP cannot wrap modules with tied parameters independently at the same level. '
                f'Please adjust the auto_wrap_policy to ensure no parameter sharing exists between modules to be sharded.',
            )

        # Check for tying between candidates and the rest of the model (using root_module);
        # As the docstring discusses, we don't allow weight sharing between fsdp and non-fsdp modules, even if the parent
        # module is not FSDP wrapped. We may consider to relax this constraint in the future.
        legalize_param_sharing_between_modules(root_module, standalone_child_candidates)

    # 3. Recurse on module's children for downstream sharding
    for child in module.children():
        _recursive_apply_fully_shard(root_module, child, target_modules_to_kwargs)

    # 4. Apply fully_shard to the module if it is in target_modules_to_kwargs
    if module in target_modules_to_kwargs:
        fully_shard(module, **target_modules_to_kwargs[module])


def apply_fully_shard(
    model: nn.Module,
    fsdp2_config: FSDP2Config,
    auto_wrap_policy: CustomPolicy,
) -> None:
    """Applies FSDP2's `fully_shard` to the specified modules and then to the parent model.

    NOTE FSDP are only applied to nn.Parameters not Buffers.

    Args:
        model (torch.nn.Module): The parent model.
        fsdp2_config (FSDP2Config): The FSDP2 configuration.
        auto_wrap_policy (CustomPolicy): The policy to apply to the model.

    Returns:
        None
    """
    # Define the default kwargs for fully_shard
    fully_shard_kwargs = {'mesh': fsdp2_config.device_mesh, 'reshard_after_forward': fsdp2_config.reshard_after_forward}

    # Get a dictionary of all submodules to wrap and their kwargs
    target_modules_to_kwargs = auto_wrap_policy._run_policy(
        root_module=model,
        ignored_modules=set(),
        root_kwargs=fully_shard_kwargs,
    )

    # Recursively apply fully_shard to each relevant submodule defined by the policy (and the corresponding target_modules_to_kwargs)
    _recursive_apply_fully_shard(model, model, target_modules_to_kwargs)


def prepare_fully_shard(
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    fsdp2_config: FSDP2Config,
    auto_wrap_policy: Optional[CustomPolicy] = None,
) -> None:
    """Applies FSDP2's `fully_shard` to the model according to given fsdp2_config.

    Args:
        model (torch.nn.Module): The model to prepare.
        fsdp2_config (FSDP2Config): The FSDP2 configuration.
        auto_wrap_policy (CustomPolicy): The policy to apply to the model.

    Returns:
        None
    """
    # Build the parameter to name mapping
    orig_param_to_name = {p: n for n, p in model.named_parameters(recurse=True)}

    # If the auto_wrap_policy is not provided, generate the default policy
    if auto_wrap_policy is None:
        auto_wrap_policy = generate_default_policy(model)

    with check_param_tying(model):
        apply_fully_shard(model, fsdp2_config, auto_wrap_policy)

    # If the optimizer is provided, update the optimizer's parameter groups to use the sharded model's DTensor parameters
    if optimizer is not None:
        update_optimizer_modules(optimizer, model, orig_param_to_name)

    # Apply activation checkpointing
    if fsdp2_config.activation_checkpointing or fsdp2_config.activation_cpu_offload:
        apply_ac(model, fsdp2_config)
