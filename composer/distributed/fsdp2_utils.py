# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Helpers for FSDP2."""

import contextlib
import warnings
from typing import Any, Callable, Union

import torch
import torch.nn as nn
from torch.distributed.fsdp import FSDPModule
from torch.distributed.fsdp.wrap import CustomPolicy
from torch.distributed.tensor import DTensor, distribute_tensor
from torchmetrics import Metric, MetricCollection

from composer.models import ComposerModel
from composer.utils.parallelism import FSDP2Config

# FSDP2 Weight Tying Functions
# TODO: These functions are all relatively similar to each other, we should consider
# refactoring them in the future to be simpler. We also might benefit from moving these
# weight tying functions to a new file (in a potential `fsdp2_utils` directory).


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

    # a naive walk over model.modules wouldn't work as if a module is filtered out, we need to skip it and its children
    # while model.modules() walk into all submodules, therefore we need to do a DFS to check for parameter sharing
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


def get_standalone_and_tied_modules(modules: list[nn.Module]) -> tuple[list[nn.Module], set[nn.Module]]:
    """Filter modules that have standalone params thus can be fully sharded independently and those with tied params.

    Note if a module is a child of another module, they are still considered to be tied modules.
    If a module does not have any params, it is not included in the output.

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

    # Return a list of sets, each set contains the FQNs for a tied parameter group
    return list(param_object_to_fqns.values())


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


# Optimizer + FSDP2 Functions


@contextlib.contextmanager
def sync_optimizer_and_model_params(
    optimizer: torch.optim.Optimizer,
    model: nn.Module,
):
    """Context manager that synchronizes optimizer parameters with model parameters.

    This context manager builds a mapping between the original model parameters and their names,
    yields control back to the caller, and then updates the optimizer's parameter groups to
    use the (potentially sharded) model parameters after the context block.

    Args:
        optimizer (torch.optim.Optimizer): The optimizer to update.
        model (nn.Module): The model whose parameters should be synced with the optimizer.

    Yields:
        None
    """
    # Build the parameter to name mapping before any modifications
    orig_param_to_name = {p: n for n, p in model.named_parameters(recurse=True)}

    yield

    # After the context, update the optimizer to use the new parameters
    update_optimizer_modules(optimizer, model, orig_param_to_name)


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
                # And since we don't have a way to identify the parameter name in the optimizer, we just use the id
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


# FSDP2 Policy Functions


def generate_default_policy(parent_model: nn.Module) -> CustomPolicy:
    """Generates the default fsdp wrap policy for FSDP2.

    This policy determines which modules should be wrapped with FSDP2 based on module attributes
    or custom wrapping functions. It checks for:

    1. The presence of an `_fsdp_wrap` attribute on modules (deprecated)
    2. A `fsdp_wrap_fn` callable on the parent model that returns True/False or FSDP2 config options

    The policy respects parameter sharing constraints, ensuring that modules with tied weights
    are properly handled during sharding.

    Args:
        parent_model (nn.Module): The root module to generate the policy for.

    Returns:
        CustomPolicy: A policy function that determines which modules to wrap with FSDP2.

    Raises:
        KeyError: If a module's fsdp_wrap_fn returns a dict with invalid FSDP2Config keys.
    """
    # Filter the specific deprecation warning to only appear once.
    warnings.filterwarnings(
        'once',
        category=DeprecationWarning,
        message='The _fsdp_wrap attribute will be removed in a future release. Please use fsdp_wrap_fn instead.',
    )

    def lambda_fn(current_module: nn.Module) -> Union[bool, dict[str, Any]]:
        if hasattr(current_module, '_fsdp_wrap'):
            warnings.warn(
                DeprecationWarning(
                    'The _fsdp_wrap attribute will be removed in a future release. Please use fsdp_wrap_fn instead.',
                ),
            )
            return bool(current_module._fsdp_wrap)
        # TODO: make this recursive for reusability, similar to meta_init in param_init.py
        if hasattr(parent_model, 'fsdp_wrap_fn') and isinstance(parent_model.fsdp_wrap_fn, Callable):
            res = parent_model.fsdp_wrap_fn(current_module)
            # Ensure all keys in the returned dict are valid FSDP2Config attributes
            if isinstance(res, dict) and not set(res.keys()).issubset(FSDP2Config.settable_attrs()):
                raise KeyError(
                    f'Invalid FSDP2 config keys in wrap_fn return value. Valid keys are: {FSDP2Config.settable_attrs()}',
                )
            return res
        return False

    return CustomPolicy(lambda_fn)


def generate_composer_model_policy(composer_model: ComposerModel) -> CustomPolicy:
    """Generates a FSDP wrap policy for ComposerModel that mimics FSDP1 behavior.

    This policy wraps all direct children of the ComposerModel but not the ComposerModel itself,
    which matches the behavior of FSDP1's prepare_fsdp_module function. It also respects
    any _fsdp_wrap attributes or fsdp_wrap_fn functions defined on modules.

    Args:
        composer_model (ComposerModel): The ComposerModel to generate a policy for.

    Returns:
        CustomPolicy: A policy function that determines which modules to wrap with FSDP.

    Raises:
        KeyError: If a module's fsdp_wrap_fn returns a dict with invalid FSDP2Config keys.
    """
    # Filter the specific deprecation warning to only appear once.
    warnings.filterwarnings(
        'once',
        category=DeprecationWarning,
        message='The _fsdp_wrap attribute will be removed in a future release. Please use fsdp_wrap_fn instead.',
    )

    cached_submodules_to_wrap: dict[nn.Module, bool | dict[str, Any]] = {composer_model: False}
    for child in composer_model.children():
        if isinstance(child, Metric | MetricCollection):
            for module in child.modules():
                cached_submodules_to_wrap[module] = False
            continue
        # this can be overwritten by the _fsdp_wrap attribute or fsdp_wrap_fn
        cached_submodules_to_wrap[child] = True
        fsdp_wrap_fn = getattr(child, 'fsdp_wrap_fn', lambda x: cached_submodules_to_wrap.get(x, False))
        for child_module in child.modules():
            if hasattr(child_module, '_fsdp_wrap'):
                warnings.warn(
                    DeprecationWarning(
                        'The _fsdp_wrap attribute will be removed in a future release. Please use fsdp_wrap_fn instead.',
                    ),
                )
                cached_submodules_to_wrap[child_module] = bool(child_module._fsdp_wrap)
            elif child_module is child:
                continue
            else:
                res = fsdp_wrap_fn(child_module)
                if isinstance(res, dict) and not set(res.keys()).issubset(FSDP2Config.settable_attrs()):
                    raise KeyError(
                        f'Invalid FSDP2 config keys in wrap_fn return value. Valid keys are: {FSDP2Config.settable_attrs()}',
                    )
                cached_submodules_to_wrap[child_module] = res

    def lambda_fn(current_module: nn.Module) -> bool | dict[str, Any]:
        return cached_submodules_to_wrap.get(current_module, False)

    return CustomPolicy(lambda_fn)


# TODO: We want to eventually use model.named_parameters(recurse=False) to get all the params
# associated with that module specifically. We need to do the following since we're following FSDP1
# conventions (and also since it's easy to support tied weights) but we want to move away from this
# approach in the future.
def _get_params_to_summon_fsdp2(module: torch.nn.Module, recurse: bool = True):
    """Gets the DTensors to materialize for an FSDP2 model based on recurse.

    If recurse=False, we can encounter the following state:
    FSDPModule_1
      |- weight (DTensor) <-- handled
      |- FSDPModule_2
      |   |- weight (DTensor)
      |- RegularModule_1
      |   |- weight (DTensor) <-- handled
      |   |- FSDPModule_3
      |   |   |- weight (DTensor)
    Where summon_full_params(FSDPModule_1) should materialize RegularModule_1.weight
    alongside the original FSDPModule_1.weight. Therefore, we use a dfs traversal
    to get all DTensors not owned by downstream FSDPModules.
    """
    dtensor_params = {}

    def _dfs(module: torch.nn.Module, prefix: str = ''):
        # Add all DTensors within this (FSDP)module
        for name, param in module.named_parameters(
            recurse=False,
            remove_duplicate=False,
        ):
            if isinstance(param, DTensor):
                full_name = f'{prefix}.{name}' if prefix else name
                dtensor_params[full_name] = param
        for child_name, child in module.named_children():
            if isinstance(child, FSDPModule) and not recurse:
                continue
            full_name = f'{prefix}.{child_name}' if prefix else child_name
            _dfs(child, full_name)

    _dfs(module, '')
    return dtensor_params


# TODO: This function only works when model is a FSDPModule and doesn't work with other parallelisms
# (like TP) which can make DTensors that are not FSDP specific. We want to support summon_full_params
# for other kinds of parallelisms so this approach might need a generalized rework in the future.
# Especially when we are able to deprecate FSDP1.
#
# Since supporting tied weights is the biggest concern, a potential approach is supporting
# taking in a dict of the model params at the start, figuring out which params are tied,
# and handling those tied params correctly for 3D parallelism and beyond.
@contextlib.contextmanager
def summon_full_params_fsdp2(
    model: torch.nn.Module,
    writeback: bool = True,
    recurse: bool = True,
    rank0_only: bool = False,
    offload_to_cpu: bool = False,
    with_grads: bool = False,
):
    """Context manager to get full params for FSDP2 models with DTensor APIs.

    Note: Although FSDP1 uses `unshard` and `reshard` for summoning full params, we use DTensor APIs
    to materialize the full parameters as that is the preferred approach for FSDP2. Additionally,
    `unshard` and `reshard` with writeback functionality is not supported for FSDP2 models.

    Writeback limitation: Only in-place modifications to parameter data are supported. The context
    manager cannot write back structural changes such as replacing a parameter with a different
    object type or setting it to None. If this occurs, the context manager will just use the original
    DTensor.

    We currently don't support rank0_only, offload_to_cpu, and with_grads.
    """
    # TODO: We want to support these arguments in the future.
    if any([rank0_only, offload_to_cpu, with_grads]):
        raise ValueError(
            'rank0_only, offload_to_cpu, and with_grads are not supported for FSDP2 models. '
            'The defaults supported are: rank0_only=False, offload_to_cpu=False, with_grads=False.',
        )

    dtensor_params = _get_params_to_summon_fsdp2(model, recurse=recurse)

    if not dtensor_params:
        yield
        return

    model_dtensors = {}
    metadata = {}
    tied_params = {}

    # We want to get the module and attr of the param, so we can assign
    # module.attr = param.full_tensor() before we yield and
    # module.attr = distributed (potentially updated) tensor after we yield.
    def _get_module_and_attr(module: torch.nn.Module, param_name: str):
        module_path, local_param_name = param_name.rsplit('.', 1)
        submodule = module.get_submodule(module_path)
        return submodule, local_param_name

    # Group parameters by their underlying tensor to handle tied parameters
    tensor_to_names = {}
    for name, dtensor_param in dtensor_params.items():
        if dtensor_param not in tensor_to_names:
            tensor_to_names[dtensor_param] = []
        tensor_to_names[dtensor_param].append(name)

    # Process parameters, handling tied parameters correctly
    # since there are cases where two regular modules share the same
    # weight within an FSDPModule (e.g. weight tied embedding layers
    # in a GPT architecture).
    processed_tensors = set()
    for name, dtensor_param in dtensor_params.items():
        metadata[name] = {
            'device_mesh': dtensor_param.device_mesh,  # type: ignore
            'placements': dtensor_param.placements,  # type: ignore
            'requires_grad': dtensor_param.requires_grad,  # type: ignore
        }
        model_dtensors[name] = dtensor_param

        # Only materialize the full tensor once per unique tensor
        if dtensor_param not in processed_tensors:
            full_tensor = dtensor_param.full_tensor()
            new_param = torch.nn.Parameter(full_tensor.detach().clone())

            # Set the same parameter instance for all tied parameters
            for tied_name in tensor_to_names[dtensor_param]:
                parent_module, attr_name = _get_module_and_attr(model, tied_name)
                setattr(parent_module, attr_name, new_param)
                tied_params[tied_name] = new_param

            processed_tensors.add(dtensor_param)

    try:
        yield
    finally:
        # Process tied parameters to ensure writeback works correctly
        processed_tensors = set()
        tensor_to_updated_dtensor = {}

        for name, dtensor_param in dtensor_params.items():
            parent_module, attr_name = _get_module_and_attr(model, name)

            if writeback and dtensor_param not in processed_tensors:
                # We update model_dtensors[name] to use the updated param
                # after the model changes. For tied parameters, we only need
                # to do this once per unique tensor.
                current_param = getattr(parent_module, attr_name)
                if hasattr(
                    current_param,
                    'data',
                ) and current_param.data is not None:
                    meta = metadata[name]
                    sharded = distribute_tensor(
                        current_param.data,
                        meta['device_mesh'],
                        meta['placements'],
                    )
                    new_param = torch.nn.Parameter(sharded)
                    new_param.requires_grad = meta['requires_grad']
                    tensor_to_updated_dtensor[dtensor_param] = new_param
                    processed_tensors.add(dtensor_param)
                else:
                    warnings.warn(
                        f'Parameter {name} cannot be written back because it has no .data attribute '
                        f'or .data is None. The original DTensor will be restored instead as structural '
                        f'changes are not supported.',
                    )

            # Restore the appropriate DTensor for this parameter
            if writeback and dtensor_param in tensor_to_updated_dtensor:
                setattr(parent_module, attr_name, tensor_to_updated_dtensor[dtensor_param])
            else:
                setattr(parent_module, attr_name, model_dtensors[name])


def validate_all_dtensors_are_fsdp_based(model: torch.nn.Module):
    """Validates that all DTensors in the model are made by a call to `fully_shard`."""
    all_params = {param for param in model.parameters() if isinstance(param, DTensor)}
    fsdp_params = set()
    for module in model.modules():
        if isinstance(module, FSDPModule):
            for param in module.parameters():
                if isinstance(param, DTensor):
                    fsdp_params.add(param)
    if all_params != fsdp_params:
        raise ValueError(
            'All DTensors in the model must be made by a call to `fully_shard`. '
            f'Found {len(all_params - fsdp_params)} DTensors that were not made by `fully_shard`.',
        )
