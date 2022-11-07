# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Helpers for running distributed data parallel training."""

import logging
from contextlib import contextmanager, nullcontext
from typing import Any, Callable, ContextManager, Dict, Optional, Sequence, Union, cast

import torch
import torch.distributed as torch_dist
from packaging import version
from torch.distributed.algorithms.ddp_comm_hooks.default_hooks import allreduce_hook
from torch.nn.parallel import DistributedDataParallel
from torchmetrics import Metric, MetricCollection

from composer.core import Precision
from composer.core.state import State
from composer.core.types import JSON
from composer.trainer.activation_checkpointing import apply_activation_checkpointing_wrapper, checkpoint_wrapper
from composer.utils import dist, ensure_tuple
from composer.utils.string_enum import StringEnum

__all__ = ['DDPSyncStrategy', 'ddp_sync_context', 'prepare_ddp_module']

log = logging.getLogger(__name__)


class DDPSyncStrategy(StringEnum):
    """How and when gradient synchronization should happen.

    Attributes:
        SINGLE_AUTO_SYNC: The default behavior. Gradients are synchronized as they
            computed, for only the final microbatch of a batch. This is the most efficient
            strategy, but can lead to errors when ``find_unused_parameters`` is set, since
            it is possible different microbatches may use different sets of parameters,
            leading to an incomplete sync.
        MULTI_AUTO_SYNC: The default behavior when ``find_unused_parameters`` is set.
            Gradients are synchronized as they are computed for all microbatches. This ensures
            complete synchronization, but is less efficient than :attr:`SINGLE_AUTO_SYNC`. This
            efficiency gap is usually small, as long as either DDP syncs are a small portion
            of the trainer's overall runtime, or the number of microbatches per batch is
            relatively small.
        FORCED_SYNC: Gradients are manually synchronized only after all gradients have been
            computed for the final microbatch of a batch. Like :attr:`MULTI_AUTO_SYNC`, this
            strategy ensures complete gradient synchronization, but this tends to be slower than
            :attr:`MULTI_AUTO_SYNC`. This is because ordinarily syncs can happen in parallel
            with the ``loss.backward()`` computation, meaning syncs can be mostly complete by
            the time that function finishes. However, in certain circumstances, syncs may take
            a very long time to complete - if there are also a lot of microbatches per batch,
            this strategy may be optimal.
    """
    SINGLE_AUTO_SYNC = 'single_auto_sync'
    MULTI_AUTO_SYNC = 'multi_auto_sync'
    FORCED_SYNC = 'forced_sync'


@contextmanager
def ddp_sync_context(state: State, is_final_microbatch: bool, sync_strategy: Union[str, DDPSyncStrategy]):
    """A context manager for handling the :class:`DDPSyncStrategy`.

    Args:
        state (State): The state of the :class:`.Trainer`.
        is_final_microbatch (bool): Whether or not the context is being used during the final
            microbatch of the gradient accumulation steps.
        sync_strategy (str | DDPSyncStrategy): The ddp sync strategy to use. If a string
            is provided, the string must be one of the values in :class:`DDPSyncStrategy`.
    """
    if not isinstance(state.model, DistributedDataParallel):
        yield
        return

    assert state.optimizers is not None, 'optimizers have not been initialized'
    sync_strategy = DDPSyncStrategy(sync_strategy)

    no_sync_context = cast(Callable[[], ContextManager], state.model.no_sync)
    auto_sync_context = nullcontext

    if sync_strategy == DDPSyncStrategy.SINGLE_AUTO_SYNC:
        context = auto_sync_context if is_final_microbatch else no_sync_context
        print ("in ddp sync context, final is: ", is_final_microbatch, "is sync auto: ", context is auto_sync_context)
        with context():
            yield

    elif sync_strategy == DDPSyncStrategy.MULTI_AUTO_SYNC:
        with auto_sync_context():
            yield

    elif sync_strategy == DDPSyncStrategy.FORCED_SYNC:
        try:
            with no_sync_context():
                yield
        finally:
            print('Exit force sync', is_final_microbatch)
            if is_final_microbatch:
                for optimizer in state.optimizers:
                    for group in optimizer.param_groups:
                        for p in group['params']:
                            if p.grad is not None:
                                dist.all_reduce(p.grad)
                                p.grad = p.grad / dist.get_world_size()

    else:
        raise ValueError('Unknown sync strategy', sync_strategy)


def prepare_ddp_module(
    module: torch.nn.Module,
    find_unused_parameters: bool,
    adaptive_gradient_accumulation: bool,
    dist_callback_obj: JSON,
) -> torch.nn.Module:
    """Wraps the module in a :class:`torch.nn.parallel.DistributedDataParallel` object if running distributed training.

    Args:
        module (torch.nn.Module): The module to wrap.
        find_unused_parameters (bool): Whether or not to do a pass over the autograd graph
            to find parameters to not expect gradients for. This is useful if there are some
            parameters in the model that are not being trained.
        adaptive_gradient_accumulation (bool): Whether adaptive gradient accumulation is enabled. If
            enabled, we require inserting a barrier ahead of gradient reduction to avoid deadlock.
        dist_callback_obj (JSON): A JSON object containing data used in the distributed callback.
    """
    if dist.is_available() and dist.is_initialized():
        if any((p.requires_grad for p in module.parameters())):
            log.debug('Wrapping model with DistributedDataParallel')
            ddp_model = DistributedDataParallel(module, find_unused_parameters=find_unused_parameters)
            if adaptive_gradient_accumulation:
                # Wrap the default reduce hook with a barrier
                ddp_model.register_comm_hook(dist_callback_obj, rank_sync_wrapper(allreduce_hook))
            return ddp_model
        return module
    if dist.is_available():
        raise RuntimeError('Please call dist.initialize_dist() before calling ddp.prepare_module()')

    raise RuntimeError('When the world size is > 1, ``torch.distributed`` must be used. However, it is '
                       'not available in your installation of PyTorch. Please install or build PyTorch '
                       'with distributed support.')


def rank_sync_wrapper(
    hook: Callable[[Any, torch_dist.GradBucket], torch.futures.Future[torch.Tensor]]
) -> Callable[[Any, torch_dist.GradBucket], torch.futures.Future[torch.Tensor]]:
    """Wrapper to insert monitored_barrier if using adaptive gradient accumulation.

    If a subset of ranks OOM, this monitored barrier fails and the error is caught so training can
    continue. Otherwise, two ranks would enter different barriers, resulting in deadlock.
    """
    
    def rank_sync_wrapper_hook(hook_state, bucket: torch_dist.GradBucket) -> torch.futures.Future[torch.Tensor]:
        print('enter sync: ', bucket.index())
        try:
            # Only put barrier in front of first bucket
            if bucket.index() == 0:
                print('Enter barrier')
                dist.barrier(group=hook_state['group'])
                print('Exit barrier')
            # Raise error because barrier in first bucket failed to go to no-op
            elif hook_state['hook_error']:
                raise RuntimeError('Timed out')
        except RuntimeError as e:
            # barrier was tripped
            if 'Timed out' in str(e):
                if bucket.index() == 0:
                    hook_state['hook_error'] = True

                def raise_timeout_error(fut):
                    del fut
                    raise e

                # Use a no-op hook and return the same gradients already on the device. If we don't
                # do the reduction, PyTorch will raise an internal error on the next backward pass
                # as the previous reduction hasn't been completed. After completing the no-op
                # reduction, re-raise the timeout error.
                fut = torch.futures.Future()
                fut.set_result(bucket.buffer())
                return fut.then(raise_timeout_error)
            else:
                raise
        print ("exiting sync")
        print ("bucket is: ", bucket.is_last(), "length of parameters is: ", len(bucket.parameters()))
        print ("length of gradients is: ", len(bucket.gradients()))
        # for param in bucket.parameters():

        return hook(hook_state['nested_state'], bucket)

    return rank_sync_wrapper_hook

def fsdp_sync_wrapper(
    hook: Callable[torch_dist.GradBucket, torch.futures.Future[torch.Tensor]]
) -> Callable[torch_dist.GradBucket, torch.futures.Future[torch.Tensor]]:
    """Wrapper to insert monitored_barrier if using adaptive gradient accumulation.

    If a subset of ranks OOM, this monitored barrier fails and the error is caught so training can
    continue. Otherwise, two ranks would enter different barriers, resulting in deadlock.
    """
    
    def fsdp_sync_wrapper_hook(hook_state, bucket: torch_dist.GradBucket) -> torch.futures.Future[torch.Tensor]:
        # print('enter sync, with bucket size: ', bucket.size())
        print ("type of bucket is: ", type(bucket))
        try:
            if hook_state['hook_error']:
                raise RuntimeError('Timed out')
            # Only put barrier in front of first bucket
            # if bucket.index() == 0:
            print('Enter barrier')
            print ('hook state is: ', hook_state['group'])
            dist.barrier(group=hook_state['group'])
            print('Exit barrier')
            # Raise error because barrier in first bucket failed to go to no-op
        except RuntimeError as e:
            # barrier was tripped
            if 'Socket Timeout' in str(e):
                # if bucket.index() == 0:
                hook_state['hook_error'] = True

                def raise_timeout_error(fut):
                    del fut
                    raise e

                # Use a no-op hook and return the same gradients already on the device. If we don't
                # do the reduction, PyTorch will raise an internal error on the next backward pass
                # as the previous reduction hasn't been completed. After completing the no-op
                # reduction, re-raise the timeout error.
                fut = torch.futures.Future()
                fut.set_result(bucket)
                # fut.set_result(bucket.buffer())
                return fut.then(raise_timeout_error)
            else:
                raise
        print ("exiting sync")

        return hook(hook_state["comm_hook_state"], bucket)

    return fsdp_sync_wrapper_hook


def get_torch_dtype(dtype: Union[Precision, str]):
    """Convert common string representations of dtypes to torch dtypes."""
    dtype = dtype.value if isinstance(dtype, Precision) else dtype
    if dtype in ['float32', 'torch.float32', 'fp32']:
        return torch.float32
    elif dtype in ['float16', 'torch.float16', 'half', 'fp16', 'amp', 'amp_fp16']:
        return torch.float16
    elif dtype in ['bfloat16', 'bfloat', 'torch.bfloat16', 'bf16', 'amp_bf16']:
        return torch.bfloat16
    else:
        raise ValueError(f'Not sure how to convert dtype={dtype} to a torch dtype.')


def prepare_fsdp_module(model: torch.nn.Module, optimizers: Optional[Union[torch.optim.Optimizer,
                                                                           Sequence[torch.optim.Optimizer]]],
                        fsdp_config: Dict[str, Any], precision: Precision,
                        dist_callback_obj: JSON,) -> None:
    """Prepare a module (assumed ComposerModel) and optimizer for use with :class:`torch.distributed.fsdp.FullyShardedDataParallel`.

    Args:
        model (torch.nn.Module): The model to wrap.
        optimizers (torch.optim.Optimizer | Sequence[torch.optim.Optimizer], optional): The optimizer for `model`, assumed to have a single param group := model.parameters().
        fsdp_config (Dict[str, Any]): The FSDP config. TODO: fill in configuration documentation
        precision: (Precision): The precision being used by the Trainer, used to fill in defaults for FSDP `mixed_precision` settings.
    """
    if version.parse(torch.__version__) < version.parse('1.12.0'):
        raise RuntimeError('To use FSDP with Composer, you must use torch>=1.12.0.')
    from torch.distributed.fsdp import (BackwardPrefetch, CPUOffload, FullyShardedDataParallel, MixedPrecision,
                                        ShardingStrategy)

    sharding_map = {
        'NO_SHARD': ShardingStrategy.NO_SHARD,
        'SHARD_GRAD_OP': ShardingStrategy.SHARD_GRAD_OP,
        'FULL_SHARD': ShardingStrategy.FULL_SHARD,
    }
    sharding_strategy = sharding_map[fsdp_config.get('sharding_strategy', 'FULL_SHARD').upper()]

    cpu_offload = CPUOffload(offload_params=True) if fsdp_config.get('cpu_offload', False) else None
    if cpu_offload is not None:
        raise ValueError('FSDP CPU Offload not supported yet.')

    mixed_precision = fsdp_config.get('mixed_precision', 'DEFAULT').upper()
    if isinstance(mixed_precision, dict):
        param_dtype = get_torch_dtype(mixed_precision.get('param_dtype', 'float32'))
        reduce_dtype = get_torch_dtype(mixed_precision.get('reduce_dtype', 'float32'))
        buffer_dtype = get_torch_dtype(mixed_precision.get('buffer_dtype', 'float32'))
    elif mixed_precision == 'FULL':
        param_dtype = torch.float32
        reduce_dtype = torch.float32
        buffer_dtype = torch.float32
    elif mixed_precision == 'DEFAULT':
        param_dtype = torch.float32
        reduce_dtype = get_torch_dtype(precision)
        buffer_dtype = torch.float32
    elif mixed_precision == 'PURE':
        param_dtype = get_torch_dtype(precision)
        reduce_dtype = get_torch_dtype(precision)
        buffer_dtype = get_torch_dtype(precision)
    else:
        raise ValueError(f'Unable to interpret mixed_precision={mixed_precision}')

    mixed_precision = MixedPrecision(
        param_dtype=param_dtype,
        reduce_dtype=reduce_dtype,
        buffer_dtype=buffer_dtype,
    )

    backward_prefetch_map = {
        'NONE': None,
        'BACKWARD_PRE': BackwardPrefetch.BACKWARD_PRE,
        'BACKWARD_POST': BackwardPrefetch.BACKWARD_POST,
    }
    backward_prefetch = backward_prefetch_map[fsdp_config.get('backward_prefetch', 'BACKWARD_POST').upper()]
    min_params = int(float(fsdp_config.get('min_params', 1e9)))
    activation_checkpointing = fsdp_config.get('activation_checkpointing', False)
    activation_cpu_offload = fsdp_config.get('activation_cpu_offload', False)

    # We choose to not wrap the ComposerModel directly, but instead wrap any submodules like `ComposerModel.model`
    # This makes it safer to call ComposerModel-specific functions like 'eval_forward' that
    # may make calls to sharded submodules. If we only wrap the submodules, then any call that ComposerModel makes
    # to a FSDP-wrapped submodule's `forward()` function will be safe and all-gather the necessary weights before `forward()`.
    for name, obj in model.named_children():
        if not isinstance(obj, (Metric, MetricCollection)):

            # If `obj` contains meta tensors, try to use `obj.param_init_fn` to initialize them
            def _param_init_fn(module: torch.nn.Module) -> None:
                module.to_empty(device=f'cuda:{torch.cuda.current_device()}')
                if hasattr(obj, 'param_init_fn') and isinstance(obj.param_init_fn, Callable):
                    module.apply(obj.param_init_fn)
                elif hasattr(module, 'reset_parameters') and isinstance(module.reset_parameters, Callable):
                    module.reset_parameters()

            # Choose which modules to FSDP wrap according to the following priority:
            # If module has attribute `module._fsdp_wrap = ...`, always respect it
            # Otherwise wrap if root object `obj.fsdp_wrap_fn(module)` is true
            # Or if unwrapped params in module in greater than or equal to fsdp_config.min_params
            def _auto_wrap_policy(module: torch.nn.Module, recurse: bool, unwrapped_params: int) -> bool:
                if recurse:
                    return True
                else:
                    if hasattr(module, '_fsdp_wrap'):
                        return bool(module._fsdp_wrap)

                    is_large = unwrapped_params >= min_params
                    if hasattr(obj, 'fsdp_wrap_fn') and isinstance(obj.fsdp_wrap_fn, Callable):
                        return obj.fsdp_wrap_fn(module) or is_large
                    else:
                        return is_large

            fsdp_obj = FullyShardedDataParallel(
                obj,
                sharding_strategy=sharding_strategy,
                auto_wrap_policy=_auto_wrap_policy,
                cpu_offload=cpu_offload,
                mixed_precision=mixed_precision,
                backward_prefetch=backward_prefetch,
                param_init_fn=_param_init_fn,
                device_id=torch.cuda.current_device(),
            )

            comm_hook = fsdp_obj._get_default_comm_hook()
            dist_callback_obj["comm_hook_state"] = fsdp_obj._get_default_comm_hook_state()
            fsdp_obj.register_comm_hook(dist_callback_obj, fsdp_sync_wrapper(comm_hook))

            # Activation Checkpointing
            if activation_checkpointing or activation_cpu_offload:
                first_wrap_fn = checkpoint_wrapper if activation_checkpointing else (lambda module: module)
                second_wrap_fn = (lambda module: checkpoint_wrapper(first_wrap_fn(module), offload_to_cpu=True)
                                 ) if activation_cpu_offload else first_wrap_fn

                # Choose which modules to activation checkpoint according to the following priority:
                # If module has attribute `module._activation_checkpointing = ...`, always respect it
                # Otherwise checkpoint if root object `obj.activation_checkpointing_fn(module)` is true
                def _check_fn(module: torch.nn.Module) -> bool:
                    if hasattr(module, '_activation_checkpointing'):
                        return bool(module._activation_checkpointing)
                    if hasattr(obj, 'activation_checkpointing_fn') and isinstance(obj.activation_checkpointing_fn,
                                                                                  Callable):
                        return obj.activation_checkpointing_fn(module)
                    return False

                apply_activation_checkpointing_wrapper(
                    fsdp_obj,
                    checkpoint_wrapper_fn=second_wrap_fn,  # type: ignore
                    check_fn=_check_fn,  # type: ignore
                )

            setattr(model, name, fsdp_obj)

    # Print FSDP wrapped model and FSDP config if `verbose=True`
    if fsdp_config.get('verbose', False):
        print(f'FSDP: Wrapped Model:')
        print(model)
        print(f'FSDP: Using sharding_strategy={sharding_strategy}')
        print(f'FSDP: Using cpu_offload={cpu_offload}')
        print(f'FSDP: Using mixed_precision={mixed_precision}')
        print(f'FSDP: Using backward_prefetch={backward_prefetch}')
        print(f'FSDP: Using min_params={min_params}')
        print(f'FSDP: Using activation_checkpointing={activation_checkpointing}')
        print(f'FSDP: Using activation_cpu_offload={activation_cpu_offload}')

    # Rebuild optimizer now that parameters are sharded
    if optimizers:
        optimizers_tuple = ensure_tuple(optimizers)
        if len(optimizers_tuple) != 1:
            raise NotImplementedError(f'Only one optimizer is supported; found {len(optimizers_tuple)} optimizers')
        optim = optimizers_tuple[0]
        optim.param_groups = []
        optim.add_param_group({'params': list(model.parameters())})
