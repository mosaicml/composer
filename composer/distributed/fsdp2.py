# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Helpers for FSDP2."""

from dataclasses import dataclass
from typing import Optional, Union

from torch import nn
from torch.distributed._tensor.device_mesh import DeviceMesh
from torch.distributed.fsdp._fully_shard import fully_shard
from torch.distributed.fsdp._fully_shard._fsdp_api import MixedPrecisionPolicy, OffloadPolicy


@dataclass
class FSDP2Config:
    """Configuration for Fully Sharded Data Parallelism (FSDP2).

    Args:
        device_mesh (Optional[DeviceMesh]): The DeviceMesh for sharding. If None, a default 1D mesh is created.
            For 1D mesh, parameters are fully sharded across the mesh (FSDP).
            For 2D mesh, parameters are sharded across the 1st dimension and replicated across the 0th dimension (HSDP).
        reshard_after_forward (Union[bool, int]): Controls parameter behavior after forward:
            - If True, reshards parameters after forward, re-all-gathers in backward.
            - If False, keeps unsharded parameters in memory, avoids all-gather in backward.
            - If int, reshards to smaller world size after forward.
            Default: True
        mp_policy (Optional[MixedPrecisionPolicy]): Mixed precision policy. Default: None
        offload_policy (Optional[OffloadPolicy]): Offloading policy. Default: None
    """
    device_mesh: Optional[DeviceMesh] = None
    reshard_after_forward: Union[bool, int] = True
    mp_policy: Optional[MixedPrecisionPolicy] = None
    offload_policy: Optional[OffloadPolicy] = None


def identify_shardable_modules(modules: list[nn.Module]) -> tuple[list[nn.Module], set[nn.Module]]:
    """Identifies modules that could be fully sharded and modules with tied parameters.

    Args:
        modules (list[torch.nn.Module]): List of modules to analyze.

    Returns:
        tuple: A tuple containing:
            - list[torch.nn.Module]: Modules that should be sharded
            - set[torch.nn.Module]: Modules with tied parameters
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
    modules_to_shard: list[nn.Module],
    fsdp2_config: FSDP2Config,
) -> None:
    """Applies FSDP2's `fully_shard` to the specified modules and then to the parent model.

    Args:
        model (torch.nn.Module): The parent model.
        modules_to_shard (list[torch.nn.Module]): The modules to apply fully_shard to.
        fsdp2_config (FSDP2Config): The FSDP2 configuration.

    Returns:
        None
    """
    fully_shard_kwargs = {'mesh': fsdp2_config.device_mesh, 'reshard_after_forward': fsdp2_config.reshard_after_forward}
    if fsdp2_config.mp_policy:
        fully_shard_kwargs['mp_policy'] = fsdp2_config.mp_policy
    if fsdp2_config.offload_policy:
        fully_shard_kwargs['offload_policy'] = fsdp2_config.offload_policy

    # Apply fully_shard to each module in the list
    if len(modules_to_shard) == 0:
        raise RuntimeError(
            "Can't find any submodules to apply FSDP, e.g., the submodules may all have tied weights. Applying FSDP to the root model does not provide any memory savings.",
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
    legalize_param_sharing_between_modules(model, modules_to_shard)

    fully_shard(modules_to_shard, **fully_shard_kwargs)
    # Apply fully_shard to the parent model to ensure all parameters are sharded
    fully_shard(model, **fully_shard_kwargs)


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
    modules_to_shard, _ = identify_shardable_modules(list(model.children()))
    apply_fully_shard(model, modules_to_shard, fsdp2_config)
