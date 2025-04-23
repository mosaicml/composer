# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Helpers for FSDP2."""


import warnings
from typing import Callable, Optional, Union

import torch
import torch.nn as nn
from torch.distributed.fsdp._fully_shard import fully_shard

from composer.utils.parallelism import FSDP2Config


def _generate_default_policy(parent_model: nn.Module) -> Callable[[nn.Module], Union[bool, dict, None]]:
    # Similar to dist_strategy.py (FSDP1 implementation)
    # The difference is that we can also return None to indicate that the module should not be wrapped,
    # but to continue checking descendants. FSDP1 does a flat scan of all submodules using .modules()
    # so we need to work around that using our recursive function.
    def lambda_fn(current_module: nn.Module) -> Union[bool, dict, None]:
        ret = None
        if hasattr(current_module, '_fsdp_wrap'):
            ret = bool(current_module._fsdp_wrap)
        elif hasattr(parent_model, 'fsdp_wrap_fn') and isinstance(parent_model.fsdp_wrap_fn, Callable):
            ret = parent_model.fsdp_wrap_fn(current_module)
            if isinstance(ret, dict):
                ret = {
                    'mesh': ret['mesh'],
                    'reshard_after_forward': ret['reshard_after_forward'],
                }
        return ret

    return lambda_fn


def _find_direct_children_to_wrap(
    module: nn.Module,
    auto_wrap_policy: Callable[[nn.Module], Union[bool, dict, None]],
) -> tuple[list[nn.Module], dict[nn.Module, dict]]:
    """Identifies direct children of a module that should be wrapped based on the policy."""
    candidates = []
    candidate_kwargs = {}
    for child in module.children():
        policy_result = auto_wrap_policy(child)
        if policy_result is True:
            candidates.append(child)
        elif isinstance(policy_result, dict):
            candidates.append(child)
            candidate_kwargs[child] = policy_result
    return candidates, candidate_kwargs


def legalize_fsdp_wrap_policy(
    module: nn.Module,
    auto_wrap_policy: Callable[[nn.Module], Union[bool, dict, None]],
) -> None:
    """Legalizes the FSDP wrap policy by ensuring that no submodule has _fsdp_wrap set to True if if the ancestor has _fsdp_wrap set to False."""
    assert auto_wrap_policy(module) is False, 'The root module must not be wrapped'
    for submodule in module.modules():
        submodule_policy = auto_wrap_policy(submodule)
        if submodule_policy is True or isinstance(submodule_policy, dict):
            raise ValueError(
                f'Submodule {submodule} has _fsdp_wrap set to True even though its ancestor {module} has _fsdp_wrap set to False. '
                f'This will cause errors with FSDP. Please adjust the auto_wrap_policy accordingly.',
            )


def _recursive_apply_fully_shard(
    module: nn.Module,
    fsdp2_config: FSDP2Config,
    auto_wrap_policy: Callable[[nn.Module], Union[bool, dict, None]],
    default_kwargs: dict,
) -> None:
    """Recursive helper to apply fully_shard based on policy and legalization."""
    # 1. Identify direct children candidates for sharding based on the policy
    child_candidates, child_candidate_kwargs = _find_direct_children_to_wrap(module, auto_wrap_policy)

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

        # Check for tying between candidates and the parent/other non-candidate children
        legalize_param_sharing_between_modules(module, standalone_child_candidates)

    # 3. Legalize all submodules for the modules where _fsdp_wrap is False (none of the descendants should be candidates for sharding)
    candidates_to_not_wrap = [child for child in module.children() if auto_wrap_policy(child) is False]
    for child in candidates_to_not_wrap:
        legalize_fsdp_wrap_policy(child, auto_wrap_policy)

    # 4. Recurse on children that were candidates for sharding or whose descendants are candidates for sharding
    recursive_candidates = [child for child in module.children() if auto_wrap_policy(child) is not False]
    for child in recursive_candidates:
        _recursive_apply_fully_shard(child, fsdp2_config, auto_wrap_policy, default_kwargs)

    # 5. Apply fully_shard to the standalone children identified earlier (Post-order application)
    for child in standalone_child_candidates:
        kwargs = child_candidate_kwargs.get(child, default_kwargs)
        fully_shard(child, **kwargs)


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
                # And as `prepare_fully_shard` takes in the optimizer itself, we don't have a way to
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
    fsdp2_config: FSDP2Config,
    auto_wrap_policy: Callable[[nn.Module], Union[bool, dict, None]],
) -> None:
    """Applies FSDP2's `fully_shard` to the specified modules and then to the parent model.

    NOTE FSDP are only applied to nn.Parameters not Buffers.

    Args:
        model (torch.nn.Module): The parent model.
        fsdp2_config (FSDP2Config): The FSDP2 configuration.
        auto_wrap_policy (Callable[[nn.Module], Union[bool, Dict, None]]): The policy to apply to the model.

    Returns:
        None
    """
    # Define the default kwargs for fully_shard
    fully_shard_kwargs = {'mesh': fsdp2_config.device_mesh, 'reshard_after_forward': fsdp2_config.reshard_after_forward}

    # Apply fully_shard to each relevant module defined by the policy
    _recursive_apply_fully_shard(model, fsdp2_config, auto_wrap_policy, fully_shard_kwargs)

    # Get the wrapping policy of the root model and apply fully_shard as specified
    root_policy_decision = auto_wrap_policy(model)

    if root_policy_decision is True or root_policy_decision is None:
        fully_shard(model, **fully_shard_kwargs)
    elif isinstance(root_policy_decision, dict):
        kwargs = root_policy_decision
        fully_shard(model, **kwargs)
    else:
        pass


def prepare_fully_shard(
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    fsdp2_config: FSDP2Config,
    auto_wrap_policy: Optional[Callable[[nn.Module], Union[bool, dict, None]]] = None,
) -> None:
    """Applies FSDP2's `fully_shard` to the model according to given fsdp2_config.

    Args:
        model (torch.nn.Module): The model to prepare.
        fsdp2_config (FSDP2Config): The FSDP2 configuration.
        auto_wrap_policy (Callable[[nn.Module], Union[bool, Dict, None]]): The policy to apply to the model.

    Returns:
        None
    """
    # Build the parameter to name mapping
    orig_param_to_name = {p: n for n, p in model.named_parameters(recurse=True)}
    
    # If the auto_wrap_policy is not provided, generate the default policy
    if auto_wrap_policy is None:
        auto_wrap_policy = _generate_default_policy(model)

    apply_fully_shard(model, fsdp2_config, auto_wrap_policy)

    # If the optimizer is provided, update the optimizer's parameter groups to use the sharded model's DTensor parameters
    if optimizer is not None:
        update_optimizer_modules(optimizer, model, orig_param_to_name)
