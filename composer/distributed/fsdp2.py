# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Helpers for FSDP2."""

from typing import Optional, Callable

import torch
import torch.nn as nn
from torch.distributed.fsdp._fully_shard import fully_shard, FSDPModule
from torch.distributed.fsdp.wrap import CustomPolicy

from composer.distributed.fsdp2_utils import (
    check_param_tying,
    generate_default_policy,
    get_standalone_and_tied_modules,
    legalize_param_sharing_between_modules,
    update_optimizer_modules,
)
from composer.utils import FSDP2Config, dist, get_device
from composer.devices import Device


def generate_oom_hook(device: Device) -> Callable:
    """Generate a hook that checks if any other rank hit an OOM.

    Here's an example of why this is needed using a simple 2-GPU setup and how it handles OOM issues during auto microbatching:

    - Rank 0: Layer 1 works fine
    - Rank 1: Layer 1 works fine
    - Rank 0: Layer 2 works fine
    - Rank 1: Layer 2 OOMs
        - Rank 1 raises an error _is_cuda_oom() [[trainer.py:2756]]
        - Rank 1 sets found_cuda_oom to 1 [[trainer.py:2758]]
        - Rank 1 creates found_cuda_oom_tensor = [1] and calls all_reduce on it with reduce_operation='MAX' [[trainer.py:2773]]
    - Rank 2: Layer 2 works fine until a hook handle is hit
        - Rank 2 sets found_cuda_oom_tensor = [0] [[fsdp2.py:72]]
        - Rank 2 calls all_reduce to set found_cuda_oom_tensor to max([0, 1]) = 1 [[fsdp2.py:73]]
        - Rank 2 sees that found_cuda_oom == 1 [[fsdp2.py:74]]
    - Rank 1:
        - Rank 1 creates all_ranks_finished_tensor = [1] and calls all_reduce on it with reduce_operation='MIN' [[trainer.py:2780]]
        - Rank 2 sees that all_ranks_finished == 0 (since rank 2 is still in mid-batch) [[trainer.py:2781]]
        - Rank 1 continues in the (while not all_ranks_finished) loop [[trainer.py:2771]]
    - Rank 2:
        - Rank 2 creates all_ranks_finished_tensor = [0] and calls all_reduce on it with reduce_operation='MIN' [[fsdp2.py:76]]
        - Rank 2 sees that all_ranks_finished == 0 (since this rank is still in the batch) [[fsdp2.py:77]]
        - Rank 2 sees that found_cuda_oom == 1, so it raises an error saying that a different rank OOMed [[fsdp2.py:80]]
    - Rank 1:
        - In the next round of the while loop, found_cuda_oom_tensor = [1] and calls all_reduce on it with reduce_operation='MAX' [[trainer.py:2773]]
    - Rank 2:
        - Rank 2 sees the error that was raised earlier (OOM on other rank) and sets found_cuda_oom to 1 [[trainer.py:2755]]
        - Rank 2 creates found_cuda_oom_tensor = [1] and calls all_reduce on it with reduce_operation='MAX' [[trainer.py:2773]]
        - As expected, found_cuda_oom == 1 [[trainer.py:2776]]
    - Rank 1:
        - Rank 1 creates all_ranks_finished_tensor = [1] (since it's in the same while loop as before) and calls all_reduce on it with reduce_operation='MIN' [[trainer.py:2780]]
        - Rank 1 sees that all_ranks_finished = 1 (as we are in the same part of the trainer code as Rank 2, Rank 2 returns the same value) [[trainer.py:2782]]
        - Rank 1 exits the while loop and adjusts the device_train_microbatch_size to half of the previous value [[trainer.py:2790]]
    - Rank 2:
        - Rank 2 creates all_ranks_finished_tensor = [1] (since it's finished the batch with an error) and calls all_reduce on it with reduce_operation='MIN' [[trainer.py:2780]]
        - Rank 2 sees that all_ranks_finished == 1 (since this rank is finished the batch) [[trainer.py:2781]]
        - Rank 2 exits the while loop and adjusts the device_train_microbatch_size to half of the previous value [[trainer.py:2790]]

    Args:
        device (torch.device): The device to check for OOM.

    Returns:
        Callable: The hook that checks if any other rank hit an OOM.
    """

    def sync_hook(*args):
        # Check if any other rank hit an OOM
        found_cuda_oom_tensor = device.tensor_to_device(torch.tensor([0], dtype=torch.uint8))
        dist.all_reduce(found_cuda_oom_tensor, reduce_operation='MAX')
        found_cuda_oom = found_cuda_oom_tensor.item()
        # Signal current rank is still in batch
        all_ranks_finished_tensor = device.tensor_to_device(torch.tensor([0], dtype=torch.uint8))
        dist.all_reduce(all_ranks_finished_tensor, reduce_operation='MIN')

        if found_cuda_oom == 1:
            raise RuntimeError('CUDA out of memory encountered on a different rank')

    return sync_hook


def add_oom_hooks(model) -> list[torch.utils.hooks.RemovableHandle]:
    """Add OOM hooks to the model and return the list of handles."""
    hook_handles = []
    hook = generate_oom_hook(get_device())
    for module in model.modules():
        if isinstance(module, FSDPModule):
            hook_handles.append(module.register_forward_pre_hook(hook, prepend=True))
            hook_handles.append(module.register_full_backward_pre_hook(hook, prepend=True))
        else:
            hook_handles.append(module.register_full_backward_hook(hook))

    return hook_handles


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
    auto_microbatching: bool = False,
) -> None:
    """Applies FSDP2's `fully_shard` to the model according to given fsdp2_config.

    Args:
        model (torch.nn.Module): The model to prepare.
        fsdp2_config (FSDP2Config): The FSDP2 configuration.
        auto_wrap_policy (CustomPolicy): The policy to apply to the model.
        auto_microbatching (bool): Whether to use auto microbatching.

    Returns:
        List[torch.utils.hooks.RemovableHandle]: A list of hook handles for the OOM hooks if auto_microbatching is enabled.
        List[str]: A list of the named modules after fully sharding.
    """
    # Build the parameter to name mapping
    orig_param_to_name = {p: n for n, p in model.named_parameters(recurse=True)}

    # If the auto_wrap_policy is not provided, generate the default policy
    if auto_wrap_policy is None:
        auto_wrap_policy = generate_default_policy(model)

    with check_param_tying(model):
        apply_fully_shard(model, fsdp2_config, auto_wrap_policy)

    # Add OOM hooks to the model
    hook_handles = []
    if auto_microbatching:
        hook_handles = add_oom_hooks(model)

    # If the optimizer is provided, update the optimizer's parameter groups to use the sharded model's DTensor parameters
    if optimizer is not None:
        update_optimizer_modules(optimizer, model, orig_param_to_name)

    # Return the same values that we expect from FSDP1 (removable handles, named modules)
    return hook_handles, model.named_modules()
