# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Helpers for FSDP2."""

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


def get_valid_modules_to_shard(model: nn.Module) -> list[nn.Module]:
    """
    Identifies modules within the root_module hierarchy that should be wrapped by FSDP,
    respecting the `_fsdp_wrap` attribute.

    - If a module has `_fsdp_wrap = True`, it is added to the list, and its descendants are not checked further (as we run into issues with tied weights).
    - If a module has `_fsdp_wrap = False`, it is not added, and its descendants are not checked further (as is the case with FSDP1)
    - If `_fsdp_wrap` is not set, the module itself is not added, but its children are recursively checked.

    Args:
        root_module (nn.Module): The root module to start the search from.

    Returns:
        list[nn.Module]: The list of modules identified for FSDP wrapping based on `_fsdp_wrap`.
            Modules are returned in depth-first traversal order.
    """
    modules_to_wrap = []
    visited_modules = set()

    def find_modules_to_wrap_recursive(module: nn.Module):
        if module in visited_modules:
            return
        visited_modules.add(module)

        fsdp_wrap_setting = getattr(module, '_fsdp_wrap', None)

        if fsdp_wrap_setting is True:
            # Found a module to wrap. Add it and stop descending this branch.
            modules_to_wrap.append(module)
            return
        elif fsdp_wrap_setting is False:
            # Explicitly told not to wrap this module or its descendants. Stop descending.
            return
        else:
            # This module isn't wrapped, check its children.
            for child_module in module.children():
                find_modules_to_wrap_recursive(child_module)

    # We start the search from the model itself
    find_modules_to_wrap_recursive(model)

    return modules_to_wrap

def prepare_fully_shard(
    model: nn.Module,
    fsdp2_config: FSDP2Config,
) -> None:
    """Applies FSDP2's `fully_shard` to the model according to given fsdp2_config.

    Args:
        model (torch.nn.Module): The model to prepare.
        fsdp2_config (FSDP2Config): The FSDP2 configuration.

    Returns:
        None
    """
    # We firstly get the modules that should be sharded given the _fsdp_wrap attribute
    modules_to_shard = get_valid_modules_to_shard(model)

    # We then filter out the modules that have tied weights
    modules_to_shard, _ = get_standalone_and_tied_modules(modules_to_shard)

    # Apply FSDP2 to the valid modules and the model itself
    apply_fully_shard(model, modules_to_shard, fsdp2_config)
