# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Helpers for FSDP2."""

import logging
from typing import Optional

import torch
import torch.nn as nn
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from torch.distributed.fsdp.wrap import CustomPolicy

from composer.distributed.fsdp2_utils import (
    check_param_tying,
    generate_default_policy,
    get_standalone_and_tied_modules,
    legalize_param_sharing_between_modules,
)
from composer.utils.parallelism import FSDP2Config

log = logging.getLogger(__name__)


def _recursive_apply_fully_shard(
    root_module: nn.Module,
    module: nn.Module,
    visited_modules: set[nn.Module],
    target_modules_to_kwargs: dict[nn.Module, dict],
) -> None:
    """Recursive helper to apply fully_shard based on policy and legalization.

    Args:
        root_module (nn.Module): The root module to check for parameter sharing.
        module (nn.Module): The current module being processed.
        visited_modules (set[nn.Module]): Set of modules that have already been visited so we don't process them twice.
        target_modules_to_kwargs (dict[nn.Module, dict]): Dictionary mapping modules to their fully_shard kwargs.

    Returns:
        None (fully_shards modules in place)
    """
    if module in visited_modules:
        return
    visited_modules.add(module)
    # 1. Identify direct children candidates for sharding based on whether they are in target_modules_to_kwargs
    child_candidates = [child for child in module.children() if child in target_modules_to_kwargs]

    # 2. Legalize child candidates
    standalone_child_candidates: list[nn.Module] = []
    if child_candidates:
        # Check for tying among the valid candidates based on the policy
        standalone_child_candidates, tied_children = get_standalone_and_tied_modules(child_candidates)
        if tied_children:
            tied_children_names = [name for name, child in module.named_children() if child in tied_children]
            raise ValueError(
                f'Detected tied parameters between modules designated for FSDP wrapping within {module}. '
                f'FSDP cannot wrap modules with tied parameters independently at the same level: '
                f'{tied_children_names}. '
                f'Please adjust the auto_wrap_policy to ensure no parameter sharing exists between modules to be sharded.',
            )

        # Check for tying between candidates and the rest of the model (using root_module);
        # As the docstring discusses, we don't allow weight sharing between fsdp and non-fsdp modules, even if the parent
        # module is not FSDP wrapped. We may consider to relax this constraint in the future.
        legalize_param_sharing_between_modules(root_module, standalone_child_candidates)

    # 3. Recurse on module's children for downstream sharding
    for child in module.children():
        _recursive_apply_fully_shard(root_module, child, visited_modules, target_modules_to_kwargs)

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
    # NOTE Model in LLM Foundry mostly uses PURE for its MP policy so we default to both bfloat16 for now
    # yet both Composer and TorchTitan's default mp_policy use bfloat16 for params all-gather and float32 for reduce-scatter
    # TODO: support user specified mp_policy
    fully_shard_kwargs = {
        'mesh': fsdp2_config.device_mesh,
        'reshard_after_forward': fsdp2_config.reshard_after_forward,
        'mp_policy': MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16),
    }

    # Get a dictionary of all submodules to wrap and their kwargs
    target_modules_to_kwargs = auto_wrap_policy._run_policy(
        root_module=model,
        ignored_modules=set(),
        root_kwargs=fully_shard_kwargs,
    )

    # Recursively apply fully_shard to each relevant submodule defined by the policy (and the corresponding target_modules_to_kwargs)
    _recursive_apply_fully_shard(model, model, set(), target_modules_to_kwargs)


def prepare_fully_shard(
    model: nn.Module,
    fsdp2_config: FSDP2Config,
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
    # If the auto_wrap_policy is not provided, generate the default policy
    if auto_wrap_policy is None:
        auto_wrap_policy = generate_default_policy(model)

    # Check for parameter tying
    with check_param_tying(model):
        apply_fully_shard(model, fsdp2_config, auto_wrap_policy)

    if fsdp2_config.verbose:
        log.info(f'FSDP2: Fully sharded model:\n{model}')
        for attr in fsdp2_config.settable_attrs():
            if attr == 'verbose':
                continue
            log.info(f'FSDP2: {attr}: {getattr(fsdp2_config, attr)}')
