import torch
import logging
from composer.core import State
from composer.utils import dist
from torch.distributed.fsdp import FullyShardedDataParallel
from torch.distributed.fsdp._runtime_utils import _post_backward_final_callback
from collections import defaultdict
from packaging import version

if version.parse(torch.__version__) >= version.parse('2.3.0'):
    from torch.amp.grad_scaler import _refresh_per_optimizer_state  # type: ignore
else:
    from torch.cuda.amp.grad_scaler import _refresh_per_optimizer_state  # type: ignore

log = logging.getLogger(__name__)

__all__ = [
    '_create_sync_hook',
    '_fsdp_reshard_and_cleanup',
    '_double_device_train_microbatch_size',
    '_closest_lower_power_of_2',
    '_clear_incomplete_train_states',
    '_found_ooms_across_ranks',
    '_update_num_consecutive_thrashes',
    '_handle_downward_search_in_automicrobatching',
    '_handle_upward_search_in_automicrobatching',
    '_handle_thrashing_in_automicrobatching'
]

def _create_sync_hook(state):
    def sync_hook(*args):
        # Check if any other rank hit an OOM
        found_cuda_oom_tensor = state.device.tensor_to_device(torch.tensor([0], dtype=torch.uint8))
        dist.all_reduce(found_cuda_oom_tensor, reduce_operation='MAX')
        found_cuda_oom = found_cuda_oom_tensor.item()
        # Signal current rank is still in batch
        all_ranks_finished_tensor = state.device.tensor_to_device(torch.tensor([0], dtype=torch.uint8))
        dist.all_reduce(all_ranks_finished_tensor, reduce_operation='MIN')
        
        if found_cuda_oom == 1:
            raise RuntimeError('CUDA out of memory encountered on a different rank')
    
    return sync_hook

def _fsdp_reshard_and_cleanup(model: torch.nn.Module):
    """Manually reshard and clean up FSDP model.

    When an exception like OOM happens, _post_backward_final_callback, which
    is registered as a backward callback, will not run. We manually call it to cleanup
    loose memory.
    """
    for __, module in model.named_modules():
        if isinstance(module, FullyShardedDataParallel):
            if module.check_is_root():
                # Only call _post_backward_final_callback on root module. It will
                # traverse and reshard all FSDP sub-modules
                _post_backward_final_callback(module, module)


def _double_device_train_microbatch_size(state: State):
    """Double device_train_microbatch_size when automicrobatching searches upward for a higher non-OOM microbatch size.

    Args:
        state (State): State of trainer.
    """
    # If any rank hit CUDA OOM, update device_train_microbatch_size and retry. Raise runtime error
    # if training 1 sample at a time still resulted in CUDA out of memory.
    assert state.device_train_microbatch_size is not None
    assert state.train_dataloader is not None

    try:
        batch_size = getattr(state.train_dataloader, 'batch_size')
    except AttributeError as e:
        # Error message when `device_train_microbatch_size` is 'auto'
        raise AttributeError(
            "`device_train_microbatch_size='auto'` requires the `state.train_dataloader` to have a `batch_size` attribute.",
        ) from e

    original_microbatch_size = state.device_train_microbatch_size
    # Device train microbatch size can't be greater than the device train batch size
    state.device_train_microbatch_size = min(int(original_microbatch_size * 2), batch_size)

def _closest_lower_power_of_2(microbatch_size: int):
    """Find the highest lower power of 2 to serve as a lower bound device_train_microbatch_size when automicrobatching 
    searches downward, due to either thrashing or when a previously non-OOMing microbatch size is now OOMing.
    Args:
        microbatch_size (int): Current device train microbatch size.
    """
    if microbatch_size <= 1:
        return 1
    return 1 << ((microbatch_size - 1).bit_length() - 1)

def _clear_incomplete_train_states(state: State):
    """Manually clear gradients when automicrobatching reruns a batch.
    Before automicrobatching tries a new higher or lower microbatch size, clear the
    training states and memory of the previous run of the batch to reset the memory to 
    before the batch was run. 
    """
    if hasattr(state, 'outputs'):
        del state.outputs
    if hasattr(state, 'loss'):
        del state.loss
    for optimizer in state.optimizers:
        optimizer.zero_grad(set_to_none=True)
    if state.scaler is not None:
        state.scaler._per_optimizer_states = defaultdict(_refresh_per_optimizer_state)
    _fsdp_reshard_and_cleanup(state.model)
    torch.cuda.empty_cache()

def _found_ooms_across_ranks(state: State, found_cuda_oom: bool):
    """Check if at least one rank, including the local rank, OOM'd in the forward/backward pass
    when using automicrobatching. This may happen when close to memory limit or with uneven memory 
    usage across ranks. 
    
    Ensure that all ranks are out of microbatch training before completing batch training or finding
    a new microbatch size. Return whether at least one rank OOM'd.
    """
    
    all_ranks_finished = False
    while not all_ranks_finished:
        # Propagate across all ranks if any rank hit CUDA OOM
        found_cuda_oom_tensor = state.device.tensor_to_device(
            torch.tensor([found_cuda_oom], dtype=torch.uint8),
        )
        dist.all_reduce(found_cuda_oom_tensor, reduce_operation='MAX')
        found_cuda_oom = found_cuda_oom_tensor.item()
        # Check if any rank is still not done with the batch. This may happen if only a
        # subset of ranks OOM, leaving some batches still in the forward pass
        all_ranks_finished_tensor = state.device.tensor_to_device(torch.tensor([1], dtype=torch.uint8))
        dist.all_reduce(all_ranks_finished_tensor, reduce_operation='MIN')
        all_ranks_finished = all_ranks_finished_tensor.item() == 1
    return found_cuda_oom

def _update_num_consecutive_thrashes(state: State, num_consecutive_thrashes: int, num_alloc_retries: int):
    """Update the number of consecutive batches where we experienced alloc retries.
    Consecutive alloc retries in GPU memory usually indicate thrashing, where GPU memory usage is so close
    to the memory limit that it hinders throughput.
    """
    # Check for alloc retries between batches
    stats = torch.cuda.memory_stats()
    cur_num_alloc_retries = stats["num_alloc_retries"]

    if cur_num_alloc_retries - num_alloc_retries > 0:
        alloc_retry_this_batch = 1
        log.info("Found alloc retries this batch: " +  str(num_alloc_retries) + " to " + str(cur_num_alloc_retries))
    else:
        alloc_retry_this_batch = 0

    # Propagate across all ranks if any rank had alloc retries this batch
    alloc_retry_tensor = state.device.tensor_to_device(
            torch.tensor([alloc_retry_this_batch], dtype=torch.uint8),
        )
    dist.all_reduce(alloc_retry_tensor, reduce_operation='MAX')
    alloc_retry_this_batch = alloc_retry_tensor.item() == 1
    if alloc_retry_this_batch:
        num_consecutive_thrashes += 1
    else:
        num_consecutive_thrashes = 0
    return num_consecutive_thrashes

def _handle_downward_search_in_automicrobatching(state: State, lowest_oom_microbatch_size: int, highest_non_oom_microbatch_size: int, lower_bound_microbatch_size: int, num_search_steps: int, max_search_steps: int):
    """Search downward for the highest non-OOMing microbatch size. 
    
    This method is only called when an OOM was seen this batch with the current state.device_train_microbatch_size.
    If this is the first time automicrobatching is searching for a non-OOMing microbatch size, or the previously highest non-OOMing power of 2 
    microbatch size is now OOMing, automicrobatching searches for the next highest power of 2 to test as a microbatch size. This resets num_search_steps
    to 1.
    Otherwise, while automicrobatching has searched for less than max_search_steps, automicrobatching binary searches downwards between the highest recorded 
    non-OOMing microbatch size and the lowest recorded OOMing microbatch size.
    Once automicrobatching has searched for max_search_steps, if the last tested microbatch size OOM'd, choose the highest previously
    recorded non-OOMing microbatch size. For the edge case where that microbatch size OOMs upon retry, binary search downward between 
    that value and lower_bound_microbatch_size, which is the highest power of 2 guaranteed to not OOM.
    """
    # Find closest lower power of 2 if previously non-OOM microbatch size is OOMing or this is the first microbatch size search
    if state.device_train_microbatch_size == lower_bound_microbatch_size: 
        lowest_oom_microbatch_size = state.device_train_microbatch_size
        lower_bound_microbatch_size = _closest_lower_power_of_2(state.device_train_microbatch_size)
        state.device_train_microbatch_size = lower_bound_microbatch_size
        highest_non_oom_microbatch_size = state.device_train_microbatch_size

        num_search_steps = 1
        # Skip return and continue searching for the highest non-OOM size in the new lower range
    else:
        if num_search_steps < max_search_steps:
            lowest_oom_microbatch_size = state.device_train_microbatch_size
            median_microbatch_size = int((lowest_oom_microbatch_size + highest_non_oom_microbatch_size) // 2)
            state.device_train_microbatch_size = median_microbatch_size

            num_search_steps += 1

            # Optimization so we don't repeat a converged value
            if lowest_oom_microbatch_size == highest_non_oom_microbatch_size:
                num_search_steps = max_search_steps + 1 # go to else protocol
                lowest_oom_microbatch_size = state.device_train_microbatch_size
                highest_non_oom_microbatch_size = lower_bound_microbatch_size
                state.device_train_microbatch_size = int((lowest_oom_microbatch_size + highest_non_oom_microbatch_size) // 2)

            # Skip return and decrease dtms, continuing the search for the highest non-OOM size
        elif num_search_steps == max_search_steps: 
            state.device_train_microbatch_size = highest_non_oom_microbatch_size

            num_search_steps += 1
            # Skip return and rerun to obtain loss - committing to this dtms unless retrying it OOMs
        else: 
            # Only end up here if a previously non-OOM microbatch size is no longer successful in the same training step, and it's not the original microbatch size

            lowest_oom_microbatch_size = state.device_train_microbatch_size
            highest_non_oom_microbatch_size = lower_bound_microbatch_size
            state.device_train_microbatch_size = int((lowest_oom_microbatch_size + highest_non_oom_microbatch_size) // 2)

            # Skip return and continue searching for the highest non-OOM size in this narrower range
    return lowest_oom_microbatch_size, highest_non_oom_microbatch_size, lower_bound_microbatch_size, num_search_steps

def _handle_upward_search_in_automicrobatching(state: State, lowest_oom_microbatch_size: int, highest_non_oom_microbatch_size: int, num_search_steps: int, max_search_steps: int):
    """Searches upward for the highest non-OOMing microbatch size. 
    
    This method is only called when the current state.device_train_microbatch_size did not OOM and automicrobatching is actively searching for a new
    microbatch size, either because this is the first search or a previously working microbatch size OOM'd.
    If the microbatch size is already equal to the batch size, automicrobatching commits to this microbatch size.
    Otherwise, while automicrobatching has searched for less than max_search_steps, automicrobatching binary searches upwards between the highest recorded 
    non-OOMing microbatch size and the lowest recorded OOMing microbatch size.
    """
    assert state.train_dataloader is not None
    try:
        batch_size = getattr(state.train_dataloader, 'batch_size')
    except AttributeError as e:
        # Error message when `device_train_microbatch_size` is 'auto'
        raise AttributeError(
            "`device_train_microbatch_size='auto'` requires the `state.train_dataloader` to have a `batch_size` attribute.",
        ) from e

    search_upwards = False

    if state.device_train_microbatch_size != batch_size:
        if num_search_steps == 0:
            highest_non_oom_microbatch_size = state.device_train_microbatch_size
            _double_device_train_microbatch_size(state)
            _clear_incomplete_train_states(state)
            search_upwards = True
        elif num_search_steps < max_search_steps: # Previous OOMs found in this training step 
            highest_non_oom_microbatch_size = state.device_train_microbatch_size
            median_microbatch_size = int((highest_non_oom_microbatch_size + lowest_oom_microbatch_size) // 2)
            state.device_train_microbatch_size = median_microbatch_size

            num_search_steps += 1

            # Optimization so we don't repeat a converged value
            if median_microbatch_size == highest_non_oom_microbatch_size:
                num_search_steps = max_search_steps

            _clear_incomplete_train_states(state)
            search_upwards = True
        # Else: reached max search steps and found a non-OOM microbatch size
    return search_upwards, highest_non_oom_microbatch_size, num_search_steps

def _handle_thrashing_in_automicrobatching(state: State):
    """Searches downward for the highest non-OOMing microbatch size that also doesn't thrash.
    This method is only called when two consecutive batches have alloc retries, indicating thrashing,
    where GPU memory usage is so close to the memory limit that it hinders throughput.
    Automicrobatching searches for the next highest power of 2 to use as the microbatch size.
    """
    lowest_oom_microbatch_size = state.device_train_microbatch_size
    lower_bound_microbatch_size = _closest_lower_power_of_2(state.device_train_microbatch_size)
    highest_non_oom_microbatch_size = lower_bound_microbatch_size
    state.device_train_microbatch_size = lower_bound_microbatch_size
    return lowest_oom_microbatch_size, highest_non_oom_microbatch_size, lower_bound_microbatch_size 
