# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Helpers for FSDP2."""

import contextlib
import warnings
from typing import Any, Callable, Optional, Union

import torch
import torch.nn as nn
from torch.distributed.fsdp._fully_shard import fully_shard
from torch.distributed.fsdp.wrap import CustomPolicy

from composer.utils.parallelism import FSDP2Config


def _generate_default_policy(parent_model: nn.Module) -> CustomPolicy:
    # The same policy as FSDP1 with some caveats around the parent_model (root_module)
    def lambda_fn(current_module: nn.Module) -> Union[bool, dict[str, Any]]:
        ret = False
        if hasattr(current_module, '_fsdp_wrap'):
            ret = bool(current_module._fsdp_wrap)
        elif hasattr(parent_model, 'fsdp_wrap_fn') and isinstance(parent_model.fsdp_wrap_fn, Callable):
            # There are certain situations where _fsdp_wrap for the parent model is not set, but we wrap submodules
            # with _fsdp_wrap_fn (e.g. wrapping all GPTBlocks). In those situations, we generally also want to wrap
            # the parent model, so we have an additional check here.
            if current_module == parent_model:
                return True
            ret = parent_model.fsdp_wrap_fn(current_module)
            if isinstance(ret, dict):
                # Ensure all keys in the returned dict are valid FSDP2Config attributes
                valid_keys = set(FSDP2Config.__annotations__.keys())
                if not set(ret.keys()).issubset(valid_keys):
                    raise ValueError(f'Invalid FSDP2 config keys in wrap_fn return value. Valid keys are: {valid_keys}')
        elif current_module == parent_model:
            # Unless the user specifically sets the _fsdp_wrap attribute to False for the parent model,
            # we default to wrapping the parent model.
            ret = True
        return ret

    return CustomPolicy(lambda_fn)


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


def _get_param_tying_groups(model: nn.Module) -> list[set[str]]:
    """Identifies groups of tied parameters within a model based on object identity.

    A parameter is considered tied if the same nn.Parameter object appears multiple
    times when iterating through model.named_parameters().

    NOTE: We take the recursive approach since .named_parameters() has a weird behavior where if you do
    m1.m2.m3.weight = m1.m2.m4.weight and then call m1.named_parameters(), it will only return the FQN for m1.m2.m3.weight
    but not m1.m2.m4.weight.
    """
    # Map parameter object to the set of FQNs associated with it
    param_object_to_fqns: dict[nn.Parameter, set[str]] = {}

    def _recursive_get_params(module: nn.Module, prefix: str = '') -> None:
        # Add parameters from current module
        for name, param in module.named_parameters(recurse=False):
            fqn = f'{prefix}.{name}' if prefix else name
            if param not in param_object_to_fqns:
                param_object_to_fqns[param] = set()
            param_object_to_fqns[param].add(fqn)

        # Recursively process child modules
        for child_name, child in module.named_children():
            child_prefix = f'{prefix}.{child_name}' if prefix else child_name
            _recursive_get_params(child, child_prefix)

    _recursive_get_params(model)

    # Filter to keep only groups where the same parameter object has multiple FQNs
    tying_groups = [fqns for fqns in param_object_to_fqns.values() if len(fqns) > 1]
    return tying_groups


@contextlib.contextmanager
def check_param_tying(model: nn.Module):
    """Context manager to verify that parameter tying relationships remain consistent.

    Checks parameter tying based on shared parameter object identity before and after the context.
    """
    pre_shard_tying_groups = _get_param_tying_groups(model)
    sorted_pre_shard_groups = sorted([sorted(group) for group in pre_shard_tying_groups])

    try:
        yield
    finally:
        post_shard_tying_groups = _get_param_tying_groups(model)
        sorted_post_shard_groups = sorted([sorted(group) for group in post_shard_tying_groups])

        if sorted_pre_shard_groups != sorted_post_shard_groups:
            raise RuntimeError(
                f'Parameter tying relationship changed during the context.\n'
                f'Pre-shard tying groups (object id): {sorted_pre_shard_groups}\n'
                f'Post-shard tying groups (object id): {sorted_post_shard_groups}',
            )


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
        auto_wrap_policy = _generate_default_policy(model)

    with check_param_tying(model):
        apply_fully_shard(model, fsdp2_config, auto_wrap_policy)

    # If the optimizer is provided, update the optimizer's parameter groups to use the sharded model's DTensor parameters
    if optimizer is not None:
        update_optimizer_modules(optimizer, model, orig_param_to_name)
