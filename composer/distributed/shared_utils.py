# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Shared utilities for distributed training."""

import functools
from typing import Callable, Optional

import torch
from packaging import version
from torch.utils.hooks import RemovableHandle
from torchmetrics import Metric, MetricCollection

from composer.devices import Device
from composer.models import ComposerModel
from composer.utils import dist, get_device


def get_valid_fsdp_module_types() -> dict[int, type]:
    """Returns a dictionary of valid FSDP module types based on the torch version.

    Returns:
        dict: Dictionary of valid FSDP module types.
    """
    from torch.distributed.fsdp import FullyShardedDataParallel
    valid_types: dict[int, type] = {1: FullyShardedDataParallel}

    if version.parse(torch.__version__) >= version.parse('2.6.0'):
        from torch.distributed.fsdp._fully_shard import FSDPModule
        valid_types[2] = FSDPModule

    return valid_types


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

    We check if other ranks OOMed after forward/backward pass when using auto microbatching. This
    may happen when close to memory limit or with uneven memory usage across ranks. Since we
    need to do this before the model weights are gathered for the next FSDP(1/2) block, we wrap every
    FSDP(1/2) block with a hook that checks if any other rank OOMed.

    Here's an example of why this is needed using a simple 2-GPU setup and how it handles OOM issues during auto microbatching.
    Note that the line numbers can be (slightly) off based on future changes made to the code.

    - Rank 0: Layer 1 works fine
    - Rank 1: Layer 1 works fine
    - Rank 0: Layer 2 OOMs
        - Rank 0 raises an error _is_cuda_oom() [[trainer.py:2756]]
        - Rank 0 sets found_cuda_oom to 1 [[trainer.py:2758]]
        - Rank 0 creates found_cuda_oom_tensor = [1] and calls all_reduce on it with reduce_operation='MAX' [[trainer.py:2773]]
    - Rank 1: Layer 2 works fine until a hook handle is hit
        - Rank 1 sets found_cuda_oom_tensor = [0] [[shared_utils.py:85]]
        - Rank 1 calls all_reduce to set found_cuda_oom_tensor to max([0, 1]) = 1 [[fsdp2.py:73]]
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


def add_fsdp_oom_hooks(model, fsdp_config_version: int, device: Optional[Device] = None) -> list[RemovableHandle]:
    """Add OOM hooks to the model and return the list of handles.

    The following sync hooks are added to prevent FSDP deadlocks that are caused when some ranks OOM
    and other ranks do not OOM, leading to OOMing ranks calling all_reduce to wait on the non-OOMing
    ranks and the non-OOMing ranks calling all_gatherbase to continue with FSDP training:

    forward_pre_hook: before forwards of FSDP modules
    full_backward_pre_hook: before backwards of FSDP modules
    full_backward_hook: before a prefetched unshard called by FSDP's `post_backward_reshard`

    View https://github.com/mosaicml/composer/pull/3510 for more details.

    Args:
        model (torch.nn.Module): The model to add the hooks to. This can be a ComposerModel and in that scenario, we need to add hooks to valid children.
        fsdp_config_version (int): The version of the FSDP config to use. This should be either 1 or 2.
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

    # In FSDP2, we don't support backward_prefetch=BACKWARD_POST, so we only need to add the OOM hooks to the
    # forward pre and backward pre hooks.
    # TODO: In FSDP1, we might not need the non-FSDP wrapped backward hook either, but we'll keep it for now until further investigation.
    # TODO: If we want to reduce as many potential deadlocks as possible, we may need to add hooks before all blocking collectives:
    #   - register_forward_pre_hook (before blocking all_gather)
    #   - register_forward_hook (before blocking reduce_scatter)
    #   - register_full_backward_pre_hook (before blocking all_gather)
    #   - register_full_backward_hook (before blocking reduce_scatter)
    # In all of these cases, some combination of no activation checkpointing/offloading, reshard_after_forward=False, or high gradient memory cost
    # could result in edge-case OOMs and deadlocks.
    fsdp_module_type = get_valid_fsdp_module_types()[fsdp_config_version]
    for root_module in root_modules_for_hooks:
        for module in root_module.modules():
            if isinstance(module, fsdp_module_type):
                hook_handles.append(module.register_forward_pre_hook(hook, prepend=True))  # type: ignore
                hook_handles.append(module.register_full_backward_pre_hook(hook, prepend=True))  # type: ignore
            elif fsdp_config_version == 1:
                hook_handles.append(module.register_full_backward_hook(hook))  # type: ignore

    return hook_handles
