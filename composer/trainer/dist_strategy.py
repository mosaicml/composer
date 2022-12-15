# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Helpers for running distributed data parallel training."""

import logging
from contextlib import contextmanager, nullcontext
from typing import Any, Callable, ContextManager, Dict, Optional, Sequence, Union, cast

import torch
from packaging import version
from torch.nn.parallel import DistributedDataParallel
from torchmetrics import Metric, MetricCollection

from composer.core import Precision
from composer.core.state import State
from composer.utils import StringEnum, dist, ensure_tuple

__all__ = ['DDPSyncStrategy', 'ddp_sync_context', 'prepare_ddp_module', 'prepare_fsdp_module']

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
            if is_final_microbatch:
                for optimizer in state.optimizers:
                    for group in optimizer.param_groups:
                        for p in group['params']:
                            if p.grad is not None:
                                dist.all_reduce(p.grad)
                                p.grad = p.grad / dist.get_world_size()

    else:
        raise ValueError('Unknown sync strategy', sync_strategy)


def prepare_ddp_module(module: torch.nn.Module, find_unused_parameters: bool) -> torch.nn.Module:
    """Wraps the module in a :class:`torch.nn.parallel.DistributedDataParallel` object if running distributed training.

    Args:
        module (torch.nn.Module): The module to wrap.
        find_unused_parameters (bool): Whether or not to do a pass over the autograd graph
            to find parameters to not expect gradients for. This is useful if there are some
            parameters in the model that are not being trained.
    """
    if dist.is_available() and dist.is_initialized():
        if any((p.requires_grad for p in module.parameters())):
            log.debug('Wrapping model with DistributedDataParallel')
            ddp_model = DistributedDataParallel(module, find_unused_parameters=find_unused_parameters)
            return ddp_model
        return module
    if dist.is_available():
        raise RuntimeError('Please call dist.initialize_dist() before calling ddp.prepare_module()')

    raise RuntimeError('When the world size is > 1, ``torch.distributed`` must be used. However, it is '
                       'not available in your installation of PyTorch. Please install or build PyTorch '
                       'with distributed support.')


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
                        fsdp_config: Dict[str, Any], precision: Precision) -> None:
    """Prepare a module (assumed ComposerModel) and optimizer for use with :class:`torch.distributed.fsdp.FullyShardedDataParallel`.

    Args:
        model (torch.nn.Module): The model to wrap.
        optimizers (torch.optim.Optimizer | Sequence[torch.optim.Optimizer], optional): The optimizer for `model`, assumed to have a single param group := model.parameters().
        fsdp_config (Dict[str, Any]): The FSDP config.
        precision: (Precision): The precision being used by the Trainer, used to fill in defaults for FSDP `mixed_precision` settings.
    """
    if version.parse(torch.__version__) < version.parse('1.13.0'):
        raise RuntimeError('To use FSDP with Composer, you must use torch>=1.13.0.')
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (apply_activation_checkpointing,
                                                                             checkpoint_wrapper)
    from torch.distributed.fsdp import (BackwardPrefetch, CPUOffload, FullyShardedDataParallel, MixedPrecision,
                                        ShardingStrategy)
    from torch.distributed.fsdp.flatten_params_wrapper import FlattenParamsWrapper

    if optimizers:
        optimizers_tuple = ensure_tuple(optimizers)
        if len(optimizers_tuple) != 1:
            raise NotImplementedError(f'Only one optimizer is supported; found {len(optimizers_tuple)} optimizers')

        # clearing optimizer param groups and state
        # that will be recreated at the end of prepare_fsdp_module
        optim = optimizers_tuple[0]
        optim.param_groups.clear()
        optim.state.clear()

    sharding_map = {
        'NO_SHARD': ShardingStrategy.NO_SHARD,
        'SHARD_GRAD_OP': ShardingStrategy.SHARD_GRAD_OP,
        'FULL_SHARD': ShardingStrategy.FULL_SHARD,
    }
    sharding_map_key = fsdp_config.get('sharding_strategy', 'FULL_SHARD').upper()
    sharding_strategy = sharding_map[sharding_map_key]

    if precision == Precision.FP32 and sharding_map_key != 'NO_SHARD':
        raise ValueError(
            f'FSDP in PyTorch 1.13 does not support precision `{precision}` with sharding_strategy `{sharding_map_key}.` '
            f'Consider using `amp` or `bf16` for precision for with sharding strategy `{sharding_map_key}.`')

    cpu_offload = CPUOffload(offload_params=True) if fsdp_config.get('cpu_offload', False) else None
    if cpu_offload is not None:
        raise ValueError('FSDP CPU Offload not supported yet.')

    mixed_precision = fsdp_config.get('mixed_precision', 'DEFAULT')
    param_dtype = None
    reduce_dtype = None
    buffer_dtype = None
    if isinstance(mixed_precision, dict):
        param_dtype = mixed_precision.get('param_dtype', None)
        if param_dtype is not None:
            param_dtype = get_torch_dtype(param_dtype)
        reduce_dtype = mixed_precision.get('reduce_dtype', None)
        if reduce_dtype is not None:
            reduce_dtype = get_torch_dtype(reduce_dtype)
        buffer_dtype = mixed_precision.get('buffer_dtype', None)
        if buffer_dtype is not None:
            buffer_dtype = get_torch_dtype(buffer_dtype)
    elif isinstance(mixed_precision, str):
        mixed_precision = mixed_precision.upper()
        if mixed_precision == 'FULL':
            pass
        elif mixed_precision == 'DEFAULT':
            reduce_dtype = get_torch_dtype(precision)
            buffer_dtype = torch.float32
        elif mixed_precision == 'PURE':
            param_dtype = get_torch_dtype(precision)
            reduce_dtype = get_torch_dtype(precision)
            buffer_dtype = get_torch_dtype(precision)
        else:
            raise ValueError(f'Unable to interpret mixed_precision={mixed_precision}')
    else:
        raise ValueError(f'Unable to interpret mixed_precision={mixed_precision}')

    if sharding_map_key != 'NO_SHARD' and (
            precision == Precision.AMP_FP16 and param_dtype not in [torch.float16, None] or
            precision == Precision.AMP_BF16 and param_dtype not in [torch.bfloat16, None]):
        raise ValueError(
            f'FSDP in PyTorch 1.13 does not support precision `{precision}` with sharding strategy `{sharding_strategy}` '
            f'and param_dtype `{param_dtype}.` Consider using one of the predefined mixed_precision strategies '
            "(choose: `'FULL'`, `'DEFAULT'`, `'PURE'`)")

    keep_low_precision_grads = fsdp_config.get('keep_low_precision_grads', False)

    mixed_precision = MixedPrecision(
        param_dtype=param_dtype,
        reduce_dtype=reduce_dtype,
        buffer_dtype=buffer_dtype,
        keep_low_precision_grads=keep_low_precision_grads,
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
    sync_module_states = fsdp_config.get('sync_module_states', False)
    forward_prefetch = fsdp_config.get('forward_prefetch', False)
    limit_all_gathers = fsdp_config.get('limit_all_gathers', False)

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
                sync_module_states=sync_module_states,
                forward_prefetch=forward_prefetch,
                limit_all_gathers=limit_all_gathers,
            )

            # Activation Checkpointing
            if activation_checkpointing or activation_cpu_offload:
                first_wrap_fn = checkpoint_wrapper if activation_checkpointing else (lambda module: module)
                second_wrap_fn = (lambda module: checkpoint_wrapper(first_wrap_fn(module), offload_to_cpu=True)
                                 ) if activation_cpu_offload else first_wrap_fn

                # Choose which modules to activation checkpoint according to the following priority:
                # If module has attribute `module._activation_checkpointing = ...`, always respect it
                # Otherwise checkpoint if root object `obj.activation_checkpointing_fn(module)` is true
                def _check_fn(module: torch.nn.Module) -> bool:
                    if isinstance(module, (FullyShardedDataParallel, FlattenParamsWrapper)):
                        return False
                    if hasattr(module, '_activation_checkpointing'):
                        return bool(module._activation_checkpointing)
                    if hasattr(obj, 'activation_checkpointing_fn') and isinstance(obj.activation_checkpointing_fn,
                                                                                  Callable):
                        return obj.activation_checkpointing_fn(module)
                    return False

                apply_activation_checkpointing(
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
        print(f'FSDP: Using sync_module_states={sync_module_states}')
        print(f'FSDP: Using forward_prefetch={forward_prefetch}')
        print(f'FSDP: Using limit_all_gathers={limit_all_gathers}')

    # Rebuild optimizer now that parameters are sharded
    if optimizers:
        optimizers_tuple = ensure_tuple(optimizers)
        optim = optimizers_tuple[0]
        optim.param_groups.clear()
        optim.add_param_group({'params': list(model.parameters())})
