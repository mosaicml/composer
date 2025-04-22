# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Helpers for FSDP2."""

import warnings
from typing import Optional

import torch
import torch.nn as nn
from torch.distributed.fsdp._fully_shard import fully_shard

from composer.utils.parallelism import FSDP2Config


def get_standalone_and_tied_modules(modules: list[nn.Module]) -> tuple[list[nn.Module], set[nn.Module]]:
    """Filter modules that have standalone params thus can be fully sharded independently and those with tied params.

    Note if a module does not have any params, it is not included in the output.

    Args:
        modules (list[torch.nn.Module]): List of modules to analyze.

    Returns:
        tuple: A tuple containing:
            - list[torch.nn.Module]: Modules that don't share parameters with other modules.
            - set[torch.nn.Module]: Modules with shared/tied parameters
    """
    # Find all tied parameters (parameters that share the same memory) between modules
    seen_params: set[nn.Parameter] = set()
    tied_params: set[nn.Parameter] = set()
    for module in modules:
        for param in module.parameters():
            if param in seen_params:
                tied_params.add(param)
        # seen_params is only updated at module granularity
        for param in module.parameters():
            seen_params.add(param)

    # if a module has tied parameters, we add it to modules_with_tied_params
    modules_with_tied_params = set()
    for module in modules:
        for param in module.parameters():
            if param in tied_params:
                modules_with_tied_params.add(module)
                break

    # Modules to shard are those that have parameters and don't have tied parameters
    modules_to_shard = [
        module for module in modules if module not in modules_with_tied_params and
        next(module.parameters(), None) is not None  # Filter out modules with no parameters
    ]

    return modules_to_shard, modules_with_tied_params


def legalize_param_sharing_between_modules(model: nn.Module, modules_to_shard: list[nn.Module]) -> None:
    """Checks if there's parameter sharing between modules to be sharded and other modules in model.

    Args:
        model (nn.Module): The root model.
        modules_to_shard (list[nn.Module]): The modules that will be sharded.

    Raises:
        ValueError: If parameter sharing is detected between modules to be sharded and other modules.
    """
    # Collect all parameters from modules to be sharded
    modules_to_shard_params = set()
    for module in modules_to_shard:
        modules_to_shard_params.update(p for p in module.parameters())

    visited_modules = set()
    modules_to_shard_set = set(modules_to_shard)

    # Define a DFS function to check for parameter sharing
    def _check_param_sharing(module: nn.Module):
        if module in modules_to_shard_set or module in visited_modules:
            return
        visited_modules.add(module)

        # Check if this module shares parameters with modules_to_shard
        for param in module.parameters(recurse=False):
            if param in modules_to_shard_params:
                raise ValueError(
                    f"Parameter sharing detected between modules to be sharded and module '{module}'. "
                    f'This will cause errors with FSDP. Either ensure no parameter sharing exists '
                    f'or include all modules with shared parameters in modules_to_shard.',
                )

        # Continue DFS with children
        for child in module.children():
            _check_param_sharing(child)

    # Start the check from the root model
    _check_param_sharing(model)


def update_optimizer_modules(
    optimizer: torch.optim.Optimizer,
    model: nn.Module,
    orig_param_to_name: dict[torch.nn.Parameter, str],
) -> None:
    """Updates the optimizer's parameter groups to use the sharded model parameters.

    Assumes no training has occurred yet and the optimizer state is empty. If the optimizer state is not empty,
    it will be cleared with a warning.

    Args:
        optimizer (Optimizer): The optimizer to update.
        modules_to_shard (list[nn.Module]): The modules that will be sharded.
        model (nn.Module): The parent model that is also sharded.
        orig_param_to_name (dict[torch.nn.Parameter, str]): Mapping from original parameters to their names.
    """
    # Check if the optimizer state is empty
    # If not, clear it and warn the user
    if optimizer.state:
        warnings.warn(
            'FSDP2 wrapping assumes the optimizer state is empty (i.e., training has not started). '
            'but non-empty optimizer state was found. Optimizer state will be cleared.',
        )
        optimizer.state.clear()

    # Build a mapping from parameter name to sharded parameter (after sharding)
    name_to_sharded_param = dict(model.named_parameters(recurse=True))

    # Create a mapping from old parameters to new DTensor parameters
    # Note: if params are tied and the same parameter is in multiple groups, pytorch will raise an error
    old_to_new_param = {}
    unseen_params = set()
    for group in optimizer.param_groups:
        for param in group['params']:
            # Note: the names of the parameters stay the same after sharding so we can do the following.
            param_name = orig_param_to_name.get(param, None)
            if param_name is None:
                # This means that the parameter is not in the original model
                # And as `apply_fully_shard` takes in the optimizer itself, we don't have a way to
                # identify the parameter name so we just use the id
                unseen_params.add(f'optimizer.param_id.{id(param)}')
            elif param_name not in name_to_sharded_param:
                # This means that the base model parameter is not in the sharded model
                # This should never happen, we note this in the error message
                unseen_params.add(f'model.param_name.{param_name}')
            else:
                old_to_new_param[param] = name_to_sharded_param[param_name]

    # Raise an error with all the parameters that were not found in the sharded model
    if len(unseen_params) > 0:
        raise ValueError(
            f'The same model must be passed to the optimizer and trainer but the '
            f'following parameters were not found in the sharded model: {list(unseen_params)}.'
            'All parameters prefixed with "optimizer.param_id" imply that the optimizer has the wrong model.'
            'All parameters prefixed with "model.param_name" imply a significant issue where sharding '
            'has not been applied correctly.',
        )

    # Update param groups with new parameters
    new_param_groups = []
    for group in optimizer.param_groups:
        new_group = {k: v for k, v in group.items() if k != 'params'}
        new_params = [old_to_new_param[param] for param in group['params']]
        new_group['params'] = new_params
        new_param_groups.append(new_group)

    # Update param groups
    optimizer.param_groups.clear()
    for group in new_param_groups:
        optimizer.add_param_group(group)


def apply_fully_shard(
    model: nn.Module,
    independent_submodules: list[nn.Module],
    fsdp2_config: FSDP2Config,
) -> None:
    """Applies FSDP2's `fully_shard` to the specified modules and then to the parent model.

    NOTE FSDP are only applied to nn.Parameters not Buffers.

    Args:
        model (torch.nn.Module): The parent model.
        independent_submodules (list[torch.nn.Module]): The modules to apply fully_shard to.
        fsdp2_config (FSDP2Config): The FSDP2 configuration.

    Returns:
        None
    """
    fully_shard_kwargs = {'mesh': fsdp2_config.device_mesh, 'reshard_after_forward': fsdp2_config.reshard_after_forward}

    # Apply fully_shard to each module in the list
    if len(independent_submodules) == 0:
        raise RuntimeError(
            "Can't find any submodules to apply FSDP, e.g., the submodules may all have tied parameters. Applying FSDP to the root model does not provide any memory savings.",
        )

    independent_submodules, modules_tied = get_standalone_and_tied_modules(independent_submodules)
    if len(modules_tied) > 0:
        raise RuntimeError(
            'Submodules to be sharded have tied parameters. FSDP cannot be applied to modules with tied parameters independently. '
            'Please ensure that the submodules do not have tied parameters.',
        )

    # NOTE there is a bug fully_shard can not handle when the model has a child module which is the child of another
    # to be FSDPed child module. For example:
    # model
    #  self.child1
    #  self.child2
    #    ├── self.child1
    #    └── grandchild
    # We can fully_shard self.child2 however if we call fully_shard on `model` again, then
    # due to `child1` is not a FSDPModule, it will try to fully_shard params in `child1` which is already sharded by `child2`,
    # and it errors out with misleading error as it thinks it is applying FSDP on top of another parallelism.

    # Currently identify_shardable_modules avoids this case through generally handling weight tying so that only parent of child1 and child2
    # is sharded. However if we allow users to call this function directly with custom modules_to_shard, we need to:
    # legalize that no module outside modules_to_shard shares parameters with modules_to_shard or
    # TODO alternatively we can fix torch/distributed/fsdp/_fully_shard/_fsdp_init.py::_get_managed_modules
    legalize_param_sharing_between_modules(model, independent_submodules)

    fully_shard(independent_submodules, **fully_shard_kwargs)
    # Apply fully_shard to the parent model to ensure all parameters are sharded
    fully_shard(model, **fully_shard_kwargs)


def prepare_fully_shard(
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    fsdp2_config: FSDP2Config,
) -> None:
    """Applies FSDP2's `fully_shard` to the model according to given fsdp2_config.

    Args:
        model (torch.nn.Module): The model to prepare.
        fsdp2_config (FSDP2Config): The FSDP2 configuration.

    Returns:
        None
    """
    # Build the parameter to name mapping
    orig_param_to_name = {p: n for n, p in model.named_parameters(recurse=True)}

    # Get the modules to shard
    modules_to_shard, _ = get_standalone_and_tied_modules(list(model.children()))

    apply_fully_shard(model, modules_to_shard, fsdp2_config)

    # If the optimizer is provided, update the optimizer's parameter groups to use the sharded model's DTensor parameters
    if optimizer is not None:
        update_optimizer_modules(optimizer, model, orig_param_to_name)
