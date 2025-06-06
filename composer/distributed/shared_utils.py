# Copyright 2025 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Shared utilities for distributed training."""

import functools
from typing import Callable, Optional

import torch
import torch.nn as nn
from torch.distributed.fsdp import FullyShardedDataParallel
from torch.utils.hooks import RemovableHandle
from torchmetrics import Metric, MetricCollection

from composer.devices import Device
from composer.models import ComposerModel
from composer.utils import dist, get_device
from composer.utils.parallelism import FSDP2Config, FSDPConfig


def get_direct_children_from_composer_model(model: ComposerModel) -> list[torch.nn.Module]:
    """Returns a list of valid direct children from a ComposerModel.

    A valid direct child for a ComposerModel is a module that's not a Metric or MetricCollection.

    Returns:
        list: List of valid direct children from a ComposerModel.
    """
    assert isinstance(model, ComposerModel)
    direct_children = []
    for child in model.children():
        if isinstance(child, (Metric, MetricCollection)):
            continue
        direct_children.append(child)

    return direct_children


def generate_oom_hook(device: Device) -> Callable:
    """Generate a hook that checks if any other rank hit an OOM.

    Note: This isn't supported for FSDP2 yet. For more details view the draft PR:
    https://github.com/mosaicml/composer/pull/3866

    We check if other ranks OOMed after forward/backward pass when using auto microbatching. This
    may happen when close to memory limit or with uneven memory usage across ranks. Since we
    need to do this before the model weights are gathered for the next FSDP1 block, we wrap every
    FSDP1 block with a hook that checks if any other rank OOMed.

    Here's an example of why this is needed using a simple 2-GPU setup and how it handles OOM issues during auto microbatching.

    Note: The line numbers below can be (slightly) off based on future changes made to the code.

    - Rank 0: Layer 1 works fine
    - Rank 1: Layer 1 works fine
    - Rank 0: Layer 2 OOMs
        - Rank 0 raises an error _is_cuda_oom() [[trainer.py:2756]]
        - Rank 0 sets found_cuda_oom to 1 [[trainer.py:2758]]
        - Rank 0 creates found_cuda_oom_tensor = [1] and calls all_reduce on it with reduce_operation='MAX' [[trainer.py:2773]]
    - Rank 1: Layer 2 works fine until a hook handle is hit
        - Rank 1 sets found_cuda_oom_tensor = [0] [[shared_utils.py:85]]
        - Rank 1 calls all_reduce to set found_cuda_oom_tensor to max([0, 1]) = 1 [[shared_utils.py:86]]
        - Rank 1 sees that found_cuda_oom == 1 [[shared_utils.py:87]]
    - Rank 0:
        - Rank 0 creates all_ranks_finished_tensor = [1] and calls all_reduce on it with reduce_operation='MIN' [[trainer.py:2780]]
        - Rank 0 sees that all_ranks_finished == 0 (since rank 1 is still in mid-batch) [[trainer.py:2781]]
        - Rank 0 continues in the (while not all_ranks_finished) loop [[trainer.py:2771]]
    - Rank 1:
        - Rank 1 creates all_ranks_finished_tensor = [0] and calls all_reduce on it with reduce_operation='MIN' [[shared_utils.py:89]]
        - Rank 1 sees that all_ranks_finished == 0 (since this rank is still in the batch) [[shared_utils.py:90]]
        - Rank 1 sees that found_cuda_oom == 1, so it raises an error saying that a different rank OOMed [[shared_utils.py:93]]
    - Rank 0:
        - In the next round of the while loop, found_cuda_oom_tensor = [1] and calls all_reduce on it with reduce_operation='MAX' [[trainer.py:2773]]
    - Rank 1:
        - Rank 1 sees the error that was raised earlier (OOM on other rank) and sets found_cuda_oom to 1 [[trainer.py:2755]]
        - Rank 1 creates found_cuda_oom_tensor = [1] and calls all_reduce on it with reduce_operation='MAX' [[trainer.py:2773]]
        - As expected, found_cuda_oom == 1 [[trainer.py:2776]]
    - Rank 0:
        - Rank 0 creates all_ranks_finished_tensor = [1] (since it's in the same while loop as before) and calls all_reduce on it with reduce_operation='MIN' [[trainer.py:2780]]
        - Rank 0 sees that all_ranks_finished = 1 (as we are in the same part of the trainer code as Rank 1, Rank 1 returns the same value) [[trainer.py:2782]]
        - Rank 0 exits the while loop and adjusts the device_train_microbatch_size to half of the previous value [[trainer.py:2790]]
    - Rank 1:
        - Rank 1 creates all_ranks_finished_tensor = [1] (since it's finished the batch with an error) and calls all_reduce on it with reduce_operation='MIN' [[trainer.py:2780]]
        - Rank 1 sees that all_ranks_finished == 1 (since this rank is finished the batch) [[trainer.py:2781]]
        - Rank 1 exits the while loop and adjusts the device_train_microbatch_size to half of the previous value [[trainer.py:2790]]

    Args:
        device (torch.device): The device to check for OOM.

    Returns:
        Callable: The hook that checks if any other rank hit an OOM.
    """

    def sync_hook(*args, device: Device):
        # Check if any other rank hit an OOM
        found_cuda_oom_tensor = device.tensor_to_device(torch.tensor([0], dtype=torch.uint8))
        dist.all_reduce(found_cuda_oom_tensor, reduce_operation='MAX')
        found_cuda_oom = found_cuda_oom_tensor.item()
        # Signal current rank is still in batch
        all_ranks_finished_tensor = device.tensor_to_device(torch.tensor([0], dtype=torch.uint8))
        dist.all_reduce(all_ranks_finished_tensor, reduce_operation='MIN')

        if found_cuda_oom == 1:
            raise RuntimeError('CUDA out of memory encountered on a different rank')

    return functools.partial(sync_hook, device=device)


def add_fsdp_oom_hooks(model: torch.nn.Module, device: Optional[Device] = None) -> list[RemovableHandle]:
    """Add OOM hooks to the FSDP1-wrapped model and return the list of handles.

    Note: This isn't supported for FSDP2 yet. For more details view the draft PR:
    https://github.com/mosaicml/composer/pull/3866

    The following sync hooks are added to prevent FSDP1 deadlocks that are caused when some ranks OOM
    and other ranks do not OOM, leading to OOMing ranks calling all_reduce to wait on the non-OOMing
    ranks and the non-OOMing ranks calling all_gatherbase to continue with FSDP training:

    forward_pre_hook: before forwards of FSDP1 modules
    full_backward_pre_hook: before backwards of FSDP1 modules
    full_backward_hook: before a prefetched unshard called by FSDP1's `post_backward_reshard`

    View https://github.com/mosaicml/composer/pull/3510 for more details.

    Args:
        model (torch.nn.Module): The model to add the hooks to. This can be a ComposerModel and in that scenario, we need to add hooks to valid children.
        device (torch.device): The device that the module is on. If None, the current rank's device will be used.

    Returns:
        list[RemovableHandle]: The list of RemovableHandles for the hooks.
    """
    hook_handles = []
    if device is None:
        device = get_device()
    hook = generate_oom_hook(device)

    # Gets the valid children if the input is a ComposerModel
    root_modules_for_hooks = []
    if isinstance(model, ComposerModel):
        root_modules_for_hooks = get_direct_children_from_composer_model(model)
    else:
        root_modules_for_hooks.append(model)

    # TODO: In FSDP1, we might not need the non-FSDP wrapped backward hook either, but we'll keep it for now until further investigation.
    # TODO: If we want to reduce as many potential deadlocks as possible, we may need to add hooks before all blocking collectives:
    #   - register_forward_pre_hook (before blocking all_gather)
    #   - register_full_backward_pre_hook (before blocking all_gather)
    #   - register_full_backward_hook (before blocking reduce_scatter)
    # In all of these cases, some combination of no activation checkpointing/offloading, reshard_after_forward=False, or high gradient memory cost
    # could result in edge-case OOMs and deadlocks.
    for root_module in root_modules_for_hooks:
        for module in root_module.modules():
            if isinstance(module, FullyShardedDataParallel):
                hook_handles.append(module.register_forward_pre_hook(hook, prepend=True))  # type: ignore
                hook_handles.append(module.register_full_backward_pre_hook(hook, prepend=True))  # type: ignore
            else:
                hook_handles.append(module.register_full_backward_hook(hook))  # type: ignore

    return hook_handles


def update_sync_module_states_if_needed(model: nn.Module, fsdp_config: FSDP2Config | FSDPConfig) -> None:
    """Updates sync_module_states configuration based on model initialization.

    In cases where the model on rank 0 is on CPU/GPU and other ranks are on meta, this function
    will automatically set sync_module_states to True. It also adds a check to make sure that
    the unexpected edge case where the model on rank 0 is on meta and other ranks are on CPU/GPU
    does not happen.

    Args:
        model (nn.Module): The model to validate.
        fsdp_config (FSDP2Config | FSDPConfig): The FSDP2 configuration containing sync_module_states setting.
    """
    device = get_device()
    requires_sync = False

    rank_on_meta = 1 if any(param.device.type == 'meta' for param in model.parameters()) else 0
    all_ranks_meta = device.tensor_to_device(torch.tensor([rank_on_meta], dtype=torch.uint8))
    dist.all_reduce(all_ranks_meta, reduce_operation='MIN')
    any_ranks_meta = device.tensor_to_device(torch.tensor([rank_on_meta], dtype=torch.uint8))
    dist.all_reduce(any_ranks_meta, reduce_operation='MAX')
    requires_sync = all_ranks_meta.item() == 0 and any_ranks_meta.item() == 1

    if not fsdp_config.sync_module_states and requires_sync:
        fsdp_config.sync_module_states = True

    # Validate that the rank setup is correct
    if fsdp_config.sync_module_states:
        expected_rank_on_meta = 0 if dist.get_global_rank() == 0 else 1
        rank_setup_valid = 1 if rank_on_meta == expected_rank_on_meta else 0

        all_ranks_valid_tensor = device.tensor_to_device(torch.tensor([rank_setup_valid], dtype=torch.uint8))
        dist.all_reduce(all_ranks_valid_tensor, reduce_operation='MIN')
        all_ranks_valid = all_ranks_valid_tensor.item()

        if all_ranks_valid == 0:
            raise ValueError(
                f'Invalid FSDP2 model initialization detected. '
                f'When doing mixed initialization, Rank 0 should have parameters on non-meta device, '
                f'and all other ranks should have parameters on meta device.',
            )
