# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Helpers for FSDP2."""

import gc
import logging
from typing import Optional

import torch
import torch.nn as nn
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from torch.distributed.fsdp.wrap import CustomPolicy
from torch.distributed.tensor import DTensor

from composer.core.precision import Precision
from composer.distributed.fsdp2_utils import (
    check_param_tying,
    generate_default_policy,
    get_standalone_and_tied_modules,
    legalize_param_sharing_between_modules,
)
from composer.distributed.mosaic_parallelism import get_mixed_precision
from composer.utils import dist
from composer.utils.parallelism import FSDP2Config

log = logging.getLogger(__name__)


def _collect_modules_for_sharding(
    module: nn.Module,
    visited_modules: set[nn.Module],
    target_modules_to_kwargs: dict[nn.Module, dict],
    modules_to_shard: list[nn.Module],
    depth: int = 0,
) -> None:
    """Recursively collect modules that need sharding in depth-first order.

    Args:
        module (nn.Module): The current module being processed.
        visited_modules (set[nn.Module]): Set of modules that have already been visited.
        target_modules_to_kwargs (dict[nn.Module, dict]): Dictionary mapping modules to their fully_shard kwargs.
        modules_to_shard (list[nn.Module]): List to collect modules in depth-first order.
        depth (int): Current depth in the module tree (for sorting).
    """
    if module in visited_modules:
        return
    visited_modules.add(module)
    
    # Recurse on children first (depth-first)
    for child in module.children():
        _collect_modules_for_sharding(child, visited_modules, target_modules_to_kwargs, modules_to_shard, depth + 1)
    
    # Add this module if it needs sharding (after children, so we get bottom-up order)
    if module in target_modules_to_kwargs:
        modules_to_shard.append((depth, module))


def _sequential_apply_fully_shard(
    root_module: nn.Module,
    target_modules_to_kwargs: dict[nn.Module, dict],
) -> None:
    """Apply fully_shard sequentially to modules in bottom-up order.

    Args:
        root_module (nn.Module): The root module to check for parameter sharing.
        target_modules_to_kwargs (dict[nn.Module, dict]): Dictionary mapping modules to their fully_shard kwargs.
    """
    from composer.distributed.prepare_distributed import log_memory_usage
    
    # Collect all modules that need sharding
    modules_to_shard: list[tuple[int, nn.Module]] = []
    visited_modules: set[nn.Module] = set()
    _collect_modules_for_sharding(root_module, visited_modules, target_modules_to_kwargs, modules_to_shard)
    
    # Sort by depth (deepest first) to ensure bottom-up processing
    modules_to_shard.sort(key=lambda x: x[0], reverse=True)
    
    log_memory_usage("Starting sequential fully_shard application")
    
    # Process each module sequentially
    for depth, module in modules_to_shard:
        module_name = module.__class__.__name__
        
        # Check for parameter tying with other modules to be sharded
        child_candidates = [child for child in module.children() if child in target_modules_to_kwargs]
        if child_candidates:
            standalone_child_candidates, tied_children = get_standalone_and_tied_modules(child_candidates)
            if tied_children:
                tied_children_names = [name for name, child in module.named_children() if child in tied_children]
                raise ValueError(
                    f'Detected tied parameters between modules designated for FSDP wrapping within {module}. '
                    f'FSDP cannot wrap modules with tied parameters independently at the same level: '
                    f'{tied_children_names}. '
                    f'Please adjust the auto_wrap_policy to ensure no parameter sharing exists between modules to be sharded.',
                )
            
            # Check for tying between candidates and the rest of the model
            legalize_param_sharing_between_modules(root_module, standalone_child_candidates)
        
        # Apply fully_shard to this module
        log_memory_usage(f"Before fully_shard for {module_name} (depth={depth})")
        fully_shard(module, **target_modules_to_kwargs[module])
        log_memory_usage(f"After fully_shard for {module_name}")
        
        # Aggressive memory cleanup after each module
        log_memory_usage(f"Before memory cleanup for {module_name}")
        gc.collect()  # First collect garbage to release Python objects
        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # Then clear the CUDA cache
        log_memory_usage(f"After memory cleanup for {module_name}")
    
    log_memory_usage("Completed sequential fully_shard application")


def apply_fully_shard(
    model: nn.Module,
    fsdp2_config: FSDP2Config,
    precision: Precision,
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
    from composer.distributed.prepare_distributed import log_memory_usage
    
    log_memory_usage("apply_fully_shard START")
    
    fully_shard_kwargs = {
        'mesh': fsdp2_config.device_mesh,
        'reshard_after_forward': fsdp2_config.reshard_after_forward,
        'mp_policy': create_mixed_precision_policy(precision, fsdp2_config.mixed_precision),
    }

    # Get a dictionary of all submodules to wrap and their kwargs
    log_memory_usage("Before auto_wrap_policy._run_policy")
    target_modules_to_kwargs = auto_wrap_policy._run_policy(
        root_module=model,
        ignored_modules=set(),
        root_kwargs=fully_shard_kwargs,
    )
    log_memory_usage("After auto_wrap_policy._run_policy")

    # Apply fully_shard sequentially to each relevant submodule defined by the policy
    log_memory_usage("Before _sequential_apply_fully_shard")
    _sequential_apply_fully_shard(model, target_modules_to_kwargs)
    log_memory_usage("After _sequential_apply_fully_shard")


def prepare_fully_shard(
    model: nn.Module,
    fsdp2_config: FSDP2Config,
    precision: Precision,
    auto_wrap_policy: Optional[CustomPolicy] = None,
) -> None:
    """Applies FSDP2's `fully_shard` to the model according to given fsdp2_config.

    Args:
        model (torch.nn.Module): The model to prepare.
        fsdp2_config (FSDP2Config): The FSDP2 configuration.
        auto_wrap_policy (Optional[CustomPolicy]): The policy to apply to the model.

    Returns:
        None
    """
    from composer.distributed.prepare_distributed import log_memory_usage
    
    log_memory_usage("prepare_fully_shard START")
    
    # If the auto_wrap_policy is not provided, generate the default policy
    if auto_wrap_policy is None:
        log_memory_usage("Before generate_default_policy")
        auto_wrap_policy = generate_default_policy(model)
        log_memory_usage("After generate_default_policy")

    # Check for parameter tying
    log_memory_usage("Before check_param_tying")
    with check_param_tying(model):
        log_memory_usage("Inside check_param_tying, before apply_fully_shard")
        apply_fully_shard(model, fsdp2_config, precision, auto_wrap_policy)
        log_memory_usage("Inside check_param_tying, after apply_fully_shard")
    log_memory_usage("After check_param_tying")

    if fsdp2_config.verbose:
        log.info(f'FSDP2: Fully sharded model:\n{model}')
        for attr in fsdp2_config.settable_attrs():
            if attr == 'verbose':
                continue
            log.info(f'FSDP2: {attr}: {getattr(fsdp2_config, attr)}')


def sync_module_states(model: nn.Module, full_state_dict: dict) -> None:
    """Syncs the module states of the model across all ranks using the full state dict.

    This function synchronizes the module states of the model across all ranks using the full state dict.
    This handles mixed initialization scenarios where different ranks have parameters on different devices.

    Args:
        model (nn.Module): The FSDP2-wrapped model to synchronize.
        full_state_dict (dict): The full state dict to synchronize. This is only fully populated on rank 0. The other ranks will receive a partial state dict.

    Returns:
        None
    """
    from torch.distributed.checkpoint.state_dict import StateDictOptions, set_model_state_dict

    # In cases where you want to FSDP2 on CPU (although not recommended)
    device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'

    if dist.get_global_rank() == 0:
        model = model.to(device=device, non_blocking=True)
    else:
        model = model.to_empty(device=device)

    options = StateDictOptions(full_state_dict=True, broadcast_from_rank0=True)

    # Sync parameters and buffers
    set_model_state_dict(model, full_state_dict, options=options)

    # Sync additional buffers that may not be in the state_dict
    for _, buffer in model.named_buffers():
        assert not isinstance(buffer, DTensor), 'Buffers should not be DTensor'
        dist.broadcast(buffer, src=0)


def create_mixed_precision_policy(precision: Precision, mixed_precision: str) -> MixedPrecisionPolicy:
    """Create a MixedPrecisionPolicy based on the precision and mixed_precision."""
    _, param_dtype, reduce_dtype, _ = get_mixed_precision(precision, mixed_precision)
    return MixedPrecisionPolicy(param_dtype=param_dtype, reduce_dtype=reduce_dtype)
