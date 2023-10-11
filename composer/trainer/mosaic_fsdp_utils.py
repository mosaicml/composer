# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

# Released under BSD 3-Clause License,
# Copyright (c) Facebook, Inc. and its affiliates.

"""Utilities for monkey patching FSDP."""

import functools
import inspect
import warnings
from functools import partial
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, Optional, Set, Tuple, Union, cast

import torch
import torch.distributed._shard.sharded_tensor.metadata as sharded_tensor_meta
import torch.nn as nn
from packaging import version
from torch import distributed
from torch.distributed import ProcessGroup
from torch.distributed._shard._utils import narrow_tensor
from torch.distributed._shard.sharded_tensor.shard import Shard
from torch.distributed._shard.sharded_tensor.utils import _parse_and_validate_remote_device
from torch.distributed._shard.sharding_spec import ChunkShardingSpec, ShardMetadata
from torch.distributed.fsdp import (BackwardPrefetch, CPUOffload, FullyShardedDataParallel, MixedPrecision,
                                    ShardingStrategy)

from composer.core import Precision
from composer.utils import dist

if TYPE_CHECKING:
    # Only include ShardedTensor when do type checking, exclude it
    # from run-time to resolve circular dependency.
    from torch.distributed._shard.sharded_tensor import ShardedTensor

SHARDING_MAP = {
    'NO_SHARD': ShardingStrategy.NO_SHARD,
    'SHARD_GRAD_OP': ShardingStrategy.SHARD_GRAD_OP,
    'FULL_SHARD': ShardingStrategy.FULL_SHARD,
}

BACKWARD_PREFETCH_MAP = {
    'NONE': None,
    'BACKWARD_PRE': BackwardPrefetch.BACKWARD_PRE,
    'BACKWARD_POST': BackwardPrefetch.BACKWARD_POST,
}


def _get_torch_dtype(dtype: Union[Precision, str]):
    """Convert common string representations of dtypes to torch dtypes."""
    dtype = dtype.value if isinstance(dtype, Precision) else dtype
    if dtype in ['float32', 'torch.float32', 'fp32']:
        return torch.float32
    elif dtype in ['float16', 'torch.float16', 'half', 'fp16', 'amp', 'amp_fp16']:
        return torch.float16
    elif dtype in ['bfloat16', 'bfloat', 'torch.bfloat16', 'bf16', 'amp_bf16']:
        return torch.bfloat16
    elif dtype in ['float8', 'torch.float8', 'fp8', 'amp_fp8']:
        if hasattr(torch, 'float8'):
            raise NotImplementedError('Torch has enabled float8. This should be updated to `return torch.float8`')
        else:
            warnings.warn('We use torch.bfloat16 by default for amp_fp8 as there is no fp8 datatype in PyTorch yet.')
            return torch.bfloat16
    else:
        raise ValueError(f'Not sure how to convert dtype={dtype} to a torch dtype.')


def get_mixed_precision(precision, mixed_precision='DEFAULT', keep_low_precision_grads=False):
    """Helper function for configuring mixed_precision."""
    param_dtype = None
    reduce_dtype = None
    buffer_dtype = None
    if isinstance(mixed_precision, dict):
        param_dtype = mixed_precision.get('param_dtype', None)
        if param_dtype is not None:
            param_dtype = _get_torch_dtype(param_dtype)
        reduce_dtype = mixed_precision.get('reduce_dtype', None)
        if reduce_dtype is not None:
            reduce_dtype = _get_torch_dtype(reduce_dtype)
        buffer_dtype = mixed_precision.get('buffer_dtype', None)
        if buffer_dtype is not None:
            buffer_dtype = _get_torch_dtype(buffer_dtype)
    elif isinstance(mixed_precision, str):
        mixed_precision = mixed_precision.upper()
        if mixed_precision == 'FULL':
            pass
        elif mixed_precision == 'DEFAULT':
            param_dtype = _get_torch_dtype(precision)
            reduce_dtype = torch.float32
            buffer_dtype = _get_torch_dtype(precision)
        elif mixed_precision == 'PURE':
            param_dtype = _get_torch_dtype(precision)
            reduce_dtype = _get_torch_dtype(precision)
            buffer_dtype = _get_torch_dtype(precision)
        else:
            raise ValueError(f'Unable to interpret mixed_precision={mixed_precision}')
    else:
        raise ValueError(f'Unable to interpret mixed_precision={mixed_precision}')

    mixed_precision = MixedPrecision(
        param_dtype=param_dtype,
        reduce_dtype=reduce_dtype,
        buffer_dtype=buffer_dtype,
        keep_low_precision_grads=keep_low_precision_grads,
    )

    return mixed_precision, param_dtype, reduce_dtype, buffer_dtype


def get_cpu_offload(cpu_offload=False):
    """Helper function for configuring cpu_offload."""
    cpu_offload = CPUOffload(offload_params=True) if cpu_offload else None
    if cpu_offload is not None:
        raise ValueError('FSDP CPU Offload not supported yet.')
    return cpu_offload


def _get_process_group(pg, process_group_cache=None):
    """Helper function for configuring and/or retrieving process groups."""
    warnings.warn(f'Instantiating FSDP with custom process groups is an experimental feature.')

    # Return regular process_groups as is, no cacheing
    if pg is None or isinstance(pg, ProcessGroup):
        return pg

    world_size = dist.get_world_size()
    local_world_size = dist.get_local_world_size()

    # Handle special str process_group cases
    if pg == 'self':
        pg = 'set1'
        warnings.warn(f"Converting process_group='self' to process_group='{pg}'")
    elif pg == 'node':
        pg = f'set{local_world_size}'
        warnings.warn(f"Converting process_group='node' to process_group='{pg}'")
    elif pg == 'local_rank_across_nodes':
        pg = f'mod{local_world_size}'
        warnings.warn(f"Converting process_group='local_rank_across_nodes' to process_group='{pg}'")

    # Handle str and Union[List[int], Tuple[int]] process_group cases
    if isinstance(pg, str) and pg.startswith('set'):
        k = int(pg.strip('set'))
        world_size = dist.get_world_size()
        if world_size % k != 0:
            raise RuntimeError(f'{world_size} must be divisible by set size ({k})')
        start = dist.get_global_rank() // k * k
        ranks = tuple(range(start, start + k))
    elif isinstance(pg, str) and pg.startswith('mod'):
        k = int(pg.strip('mod'))
        world_size = dist.get_world_size()
        if world_size % k != 0:
            raise RuntimeError(f'{world_size} must be divisible by mod ({k})')
        ranks = tuple(range(dist.get_global_rank() % k, world_size, k))
    elif isinstance(pg, (list, tuple)):
        ranks = tuple(pg)
    else:
        raise ValueError(f'Unsure how to setup process_group={pg}')

    if process_group_cache is not None and ranks in process_group_cache:
        warnings.warn(
            f'On rank={dist.get_global_rank()} using cached progress group with {ranks=}. ' +
            'If the intention was to use a new process group, a new process group can be instantiated and passed' +
            " in as an arguement (`'process_group': newly_instantiated_process_group_obect,`)")
        return process_group_cache[ranks]

    warnings.warn(
        f'Composer is instantiating custom process groups with {ranks=} (on rank={dist.get_global_rank()}). ' +
        'This is an experimental feature.')

    ranks_per_subgroup_list = list(set(dist.all_gather_object(ranks)))
    (
        current_group,
        _subgroups,
    ) = distributed.distributed_c10d.new_subgroups_by_enumeration(ranks_per_subgroup_list)

    if process_group_cache is not None:
        process_group_cache[ranks] = current_group
    return current_group


def _set_custom_fsdp_module_kwargs(module_kwargs: Dict, process_group_cache: Dict[Tuple[int], Any]) -> Dict:
    """Set custom module_kwargs per fsdp module."""
    if ('sharding_strategy' in module_kwargs and module_kwargs['sharding_strategy'] not in SHARDING_MAP.values()):
        module_kwargs['sharding_strategy'] = SHARDING_MAP[module_kwargs['sharding_strategy'].upper()]
    if 'backward_prefetch' in module_kwargs:
        if module_kwargs['backward_prefetch'] not in BACKWARD_PREFETCH_MAP.values():
            module_kwargs['backward_prefetch'] = BACKWARD_PREFETCH_MAP[module_kwargs['backward_prefetch'].upper()]
    if 'cpu_offload' in module_kwargs and not isinstance(module_kwargs['cpu_offload'], CPUOffload):
        module_kwargs['cpu_offload'] = get_cpu_offload(cpu_offload=module_kwargs['cpu_offload'].upper())
    if 'mixed_precision' in module_kwargs and not isinstance(module_kwargs['mixed_precision'], MixedPrecision):
        # `precision` needs to set `'mixed_precision'`, but `precision` is not part of fsdp kwargs
        raise NotImplementedError(
            f"Automated setting of custom per module mixed_precision is not implemented, but it can be set if `isinstance(module_kwargs['mixed_precision'], MixedPrecision)`"
        )
    if 'process_group' in module_kwargs:
        module_kwargs['process_group'] = _get_process_group(module_kwargs['process_group'], process_group_cache)

    return module_kwargs


def _custom_recursive_wrap_t1p13p1(
    module: nn.Module,
    auto_wrap_policy: Callable,
    wrapper_cls: Callable,
    ignored_modules: Set[nn.Module],
    ignored_params: Set[nn.Parameter],
    process_group_cache: Dict[Tuple[int], Any],
    only_wrap_children: bool = False,
    **kwargs: Any,
) -> Tuple[nn.Module, int]:
    """Updates FSDPs _recursive_wrap to enable module_kwargs and custom process_group cache.

    torch version must be 1.13.1.

    modified version of
    https://github.com/pytorch/pytorch/blob/d922c29a22e4bf0fba49526f7536395eb8cd66f4/torch/distributed/fsdp/wrap.py#L353
    which recursively wraps modules as FSDP modules for parameter sharding.
    This modification enables the user to pass custom FSDP arguements for every wrapped module.
    The added process_group_cache enables different FSDP modules to, when appropriate, use the
    same process group instead of instantiating a new process group.

    Automatically wrap child modules of *module* that meet the given
    criteria with :func:`auto_wrap`. Does not rely on _ConfigAutoWrap.

    Args:
        module (nn.Module):
            module to recursively wrap
        auto_wrap_policy (Callable):
            A callable specifying a policy to recursively wrap layers with FSDP.
        ignored_modules (Set[torch.nn.Module]): Modules to ignore when
            wrapping.
        ignored_params (Set[torch.nn.Parameter]): Parameters to ignore when
            wrapping; these should be the parameters contained in the modules
            in ``ignored_modules``.
        process_group_cache (Dict[Tuple[int], Any]): a cache of process_group to
            use instead of potentially instantiating a new process_group

    Returns:
        (nn.Module, int):
            Wrapped module and the number parameters wrapped recursively.
    """
    from torch.distributed.fsdp.wrap import _wrap

    assert auto_wrap_policy is not None, 'Must specify auto_wrap_policy.'
    assert wrapper_cls is not None, 'Must specify wrapper_cls'
    # Make sure no child is already wrapped.
    for _, child in module.named_modules():
        if child in ignored_modules:
            continue
        try:
            assert not isinstance(child, cast(type, wrapper_cls))
        except TypeError:
            # wrapper_cls is a function as opposed to a class type, just bypass above check.
            pass

    # We count all params, assuming none of them are already wrapped.
    num_params = sum(p.numel() for p in module.parameters() if p not in ignored_params)

    assert auto_wrap_policy is not None
    if auto_wrap_policy(module=module, recurse=True, unwrapped_params=num_params):
        total_wrapped_params = 0
        # Iterate through the children, recursively wrap if necessary
        for name, child in module.named_children():
            if child in ignored_modules:
                continue
            wrapped_child, num_wrapped_params = _custom_recursive_wrap_t1p13p1(
                module=child,
                auto_wrap_policy=auto_wrap_policy,
                wrapper_cls=wrapper_cls,
                ignored_modules=ignored_modules,
                ignored_params=ignored_params,
                process_group_cache=process_group_cache,
                **kwargs,
            )
            setattr(module, name, wrapped_child)
            # Keep track of how many parameters have been wrapped
            total_wrapped_params += num_wrapped_params
        # decide if we need to wrap the current module,
        # since the left over parameters exceed the number of params to wrap
        remainder = num_params - total_wrapped_params
        module_kwargs = auto_wrap_policy(module=module, recurse=False, unwrapped_params=remainder)
        if not only_wrap_children and module_kwargs:
            # CHANGE: We modify the original code to support custom FSDP kwargs and add
            # the process_group_cache to avoid instantiating a new process group.
            module_kwargs = module_kwargs if isinstance(module_kwargs, dict) else {}
            module_kwargs = _set_custom_fsdp_module_kwargs(module_kwargs, process_group_cache)

            final_kwargs = {**kwargs, **module_kwargs}

            # Leaf node or final wrapping of the remainder both happen here.
            return _wrap(module, wrapper_cls, **final_kwargs), num_params
        else:
            return module, total_wrapped_params
    return module, 0


def custom_auto_wrap_t1p13p1(
    self,
    auto_wrap_kwargs: Dict[str, Any],
    fsdp_kwargs: Dict[str, Any],
) -> None:
    """Updates _auto_wrap to enable module_kwargs.

    torch version must be 1.13.1.

    modified version of
    https://github.com/pytorch/pytorch/blob/d922c29a22e4bf0fba49526f7536395eb8cd66f4/torch/distributed/fsdp/fully_sharded_data_parallel.py#L1252
    FSDP's _auto_wrap recursively wraps modules as FSDP modules for parameter sharding.
    This modification enables the user to pass custom FSDP arguements for every wrapped module.
    The added process_group_cache enables different FSDP modules to, when appropriate, use the
    same process group instead of instantiating a new process group.

    Recursively auto wraps the root module given by the key "module" in
    ``auto_wrap_kwargs`` with the arguments in ``auto_wrap_kwargs`` and
    ``fsdp_kwargs``.
    Precondition: ``auto_wrap_policy`` contains the arguments expected by
    ``_recursive_wrap()``, where ``auto_wrap_policy`` is not ``None``.
    ``fsdp_kwargs`` contains all FSDP arguments except ``module``.
    """
    from torch.distributed.fsdp._utils import _contains_batchnorm, _override_batchnorm_mixed_precision
    from torch.distributed.fsdp.wrap import _or_policy, _wrap_batchnorm_individually

    auto_wrap_policy = auto_wrap_kwargs['auto_wrap_policy']
    root_module = auto_wrap_kwargs['module']
    assert auto_wrap_policy is not None
    # For auto wrapping, submodules should not already be wrapped with FSDP
    # since double wrapping is not supported
    for module_name, module in root_module.named_modules():
        if isinstance(module, FullyShardedDataParallel):
            raise ValueError(f'Expected {module_name} to NOT be FullyShardedDataParallel '
                             'if using an `auto_wrap_policy`')
    mixed_precision = fsdp_kwargs['mixed_precision']
    if mixed_precision is not None and _contains_batchnorm(root_module):
        _override_batchnorm_mixed_precision(root_module)
        auto_wrap_policy = functools.partial(_or_policy, policies=[_wrap_batchnorm_individually, auto_wrap_policy])
        warnings.warn('Both mixed precision and an `auto_wrap_policy` were specified '
                      'for FSDP, where the wrapped module has batch norm submodules. '
                      'The batch norm submodules will be wrapped as separate FSDP '
                      'instances with mixed precision disabled since some batch norm '
                      'kernels do not support low precision.')
        auto_wrap_kwargs['auto_wrap_policy'] = auto_wrap_policy
    # CHANGE: Add process group cache and call our custom _recursive_wrap
    auto_wrap_kwargs['process_group_cache'] = {}
    _custom_recursive_wrap_t1p13p1(**auto_wrap_kwargs, **fsdp_kwargs)


def _custom_recursive_wrap_t2p0p1(
    module: nn.Module,
    auto_wrap_policy: Callable,
    wrapper_cls: Callable,
    ignored_modules: Set[nn.Module],
    ignored_params: Set[nn.Parameter],
    process_group_cache: Dict[Tuple[int], Any],
    only_wrap_children: bool = False,
    **kwargs: Any,
) -> Tuple[nn.Module, int]:
    """Updates FSDPs _recursive_wrap to enable module_kwargs and custom process_group cache.

    torch version must be 2.0.1.

    modified version of
    https://github.com/pytorch/pytorch/blob/96ca226a7332be0d8f3d6159d0c797e032ab0721/torch/distributed/fsdp/wrap.py#L320
    which recursively wraps modules as FSDP modules for parameter sharding.
    This modification enables the user to pass custom FSDP arguements for every wrapped module.
    The added process_group_cache enables different FSDP modules to, when appropriate, use the
    same process group instead of instantiating a new process group.

    Wraps submodules of ``module`` for which ``auto_wrap_policy`` returns
    ``True`` with ``wrapper_cls``.

    Args:
        module (nn.Module): Module to recursively wrap.
        auto_wrap_policy (Callable): A callable representing a policy that
            determines which modules to recursively wrap with ``wrapper_cls``.
        wrapper_cls: wrapper_cls
        ignored_modules (Set[torch.nn.Module]): Modules to ignore when
            wrapping.
        ignored_params (Set[torch.nn.Parameter]): Parameters to ignore when
            wrapping; these should be the parameters contained in the modules
            in ``ignored_modules``.
        process_group_cache (Dict[Tuple[int], Any]): a cache of process_group to
            use instead of potentially instantiating a new process_group
        only_wrap_children: warp only children
    Returns:
        (nn.Module, int):
            ``module`` after wrapping and the numel recursively wrapped.
    """
    from torch.distributed.fsdp.wrap import _wrap

    assert auto_wrap_policy is not None, 'Must specify auto_wrap_policy.'
    assert wrapper_cls is not None, 'Must specify wrapper_cls'
    # Make sure no child is already wrapped.
    for _, child in module.named_modules():
        if child in ignored_modules:
            continue
        try:
            assert not isinstance(child, cast(type, wrapper_cls))
        except TypeError:
            # wrapper_cls is a function as opposed to a class type, just bypass above check.
            pass

    # We count all params, assuming none of them are already wrapped.
    nonwrapped_numel = sum(p.numel() for p in module.parameters() if p not in ignored_params)

    assert auto_wrap_policy is not None
    if auto_wrap_policy(module=module, recurse=True, nonwrapped_numel=nonwrapped_numel):
        total_wrapped_numel = 0
        # Iterate through the children, recursively wrap if necessary
        for name, child in module.named_children():
            if child in ignored_modules:
                continue
            wrapped_child, num_wrapped_params = _custom_recursive_wrap_t2p0p1(
                module=child,
                auto_wrap_policy=auto_wrap_policy,
                wrapper_cls=wrapper_cls,
                ignored_modules=ignored_modules,
                ignored_params=ignored_params,
                process_group_cache=process_group_cache,
                **kwargs,
            )
            setattr(module, name, wrapped_child)
            # Keep track of how many parameters have been wrapped
            total_wrapped_numel += num_wrapped_params
        # decide if we need to wrap the current module,
        # since the left over parameters exceed the number of params to wrap
        remainder = nonwrapped_numel - total_wrapped_numel
        module_kwargs = auto_wrap_policy(module=module, recurse=False, nonwrapped_numel=remainder)
        if not only_wrap_children and module_kwargs:
            # CHANGE: We modify the original code to support custom FSDP kwargs and add
            # the process_group_cache to avoid instantiating a new process group.
            module_kwargs = module_kwargs if isinstance(module_kwargs, dict) else {}
            module_kwargs = _set_custom_fsdp_module_kwargs(module_kwargs, process_group_cache)

            final_kwargs = {**kwargs, **module_kwargs}

            if final_kwargs.get('process_group', None) is not None:
                _pg_ranks = distributed.get_process_group_ranks(final_kwargs['process_group'])
                _meta_init = any(p.device.type == 'meta' for p in module.parameters())
                if (_meta_init and len(_pg_ranks) != dist.get_world_size() and final_kwargs.get('use_orig_params')):
                    raise NotImplementedError(
                        f'FSDP with custom process groups cannot use `use_orig_params: True` when using meta init.')

            # Leaf node or final wrapping of the remainder both happen here.
            return _wrap(module, wrapper_cls, **final_kwargs), nonwrapped_numel
        else:
            return module, total_wrapped_numel
    return module, 0


def _custom_auto_wrap_t2p0p1(
        auto_wrap_kwargs: Dict[str, Any],
        fsdp_kwargs: Dict[str, Any],
        module_wrapper_cls: Any,  # e.g. `FullyShardedDataParallel`
) -> None:
    """Updates _auto_wrap to enable module_kwargs.

    torch version must be 2.0.1.

    modified version of
    https://github.com/pytorch/pytorch/blob/96ca226a7332be0d8f3d6159d0c797e032ab0721/torch/distributed/fsdp/_wrap_utils.py#L31
    FSDP's _auto_wrap recursively wraps modules as FSDP modules for parameter sharding.
    This modification enables the user to pass custom FSDP arguements for every wrapped module.
    The added process_group_cache enables different FSDP modules to, when appropriate, use the
    same process group instead of instantiating a new process group.

    Recursively auto wraps the root module given by the key "module" in
    ``auto_wrap_kwargs`` with the arguments in ``auto_wrap_kwargs`` and
    ``fsdp_kwargs``.

    Precondition: ``auto_wrap_policy`` contains the arguments expected by
    ``_recursive_wrap()``, where ``auto_wrap_policy`` is not ``None``.
    ``fsdp_kwargs`` contains all FSDP arguments except ``module``.
    """
    from torch.distributed.fsdp._utils import _contains_batchnorm, _override_batchnorm_mixed_precision
    from torch.distributed.fsdp.wrap import _FSDPPolicy, _or_policy, _wrap_batchnorm_individually

    auto_wrap_policy = auto_wrap_kwargs['auto_wrap_policy']
    # Support new way to pass an auto wrap policy
    if isinstance(auto_wrap_policy, _FSDPPolicy):
        auto_wrap_policy = auto_wrap_policy.policy
    root_module = auto_wrap_kwargs['module']
    assert auto_wrap_policy is not None
    # For auto wrapping, submodules should not already be wrapped with FSDP
    # since double wrapping is not supported
    for module_name, module in root_module.named_modules():
        if isinstance(module, module_wrapper_cls):
            raise ValueError(f'Expected {module_name} to NOT be FullyShardedDataParallel '
                             'if using an `auto_wrap_policy`')
    mixed_precision = fsdp_kwargs['mixed_precision']
    if mixed_precision is not None and _contains_batchnorm(root_module):
        _override_batchnorm_mixed_precision(root_module)
        auto_wrap_policy = functools.partial(_or_policy, policies=[_wrap_batchnorm_individually, auto_wrap_policy])
        warnings.warn('Both mixed precision and an `auto_wrap_policy` were specified '
                      'for FSDP, where the wrapped module has batch norm submodules. '
                      'The batch norm submodules will be wrapped as separate FSDP '
                      'instances with mixed precision disabled since some batch norm '
                      'kernels do not support low precision.')
    auto_wrap_kwargs['auto_wrap_policy'] = auto_wrap_policy

    # CHANGE: Add process group cache and call our custom _recursive_wrap
    auto_wrap_kwargs['process_group_cache'] = {}
    _custom_recursive_wrap_t2p0p1(**auto_wrap_kwargs, **fsdp_kwargs)


if version.parse(torch.__version__) >= version.parse('2.0.1') and version.parse(
        torch.__version__) < version.parse('2.0.2'):
    from torch.distributed.fsdp._init_utils import ProcessGroupType
    from torch.distributed.fsdp.wrap import _FSDPPolicy

    def init_fn_t2p0p1(
        self,
        module: nn.Module,
        process_group: ProcessGroupType = None,
        sharding_strategy: Optional[ShardingStrategy] = None,
        cpu_offload: Optional[CPUOffload] = None,
        auto_wrap_policy: Optional[Union[Callable, _FSDPPolicy]] = None,
        backward_prefetch: Optional[BackwardPrefetch] = BackwardPrefetch.BACKWARD_PRE,
        mixed_precision: Optional[MixedPrecision] = None,
        ignored_modules: Optional[Iterable[torch.nn.Module]] = None,
        param_init_fn: Optional[Callable[[nn.Module], None]] = None,
        device_id: Optional[Union[int, torch.device]] = None,
        sync_module_states: bool = False,
        forward_prefetch: bool = False,
        limit_all_gathers: bool = False,
        use_orig_params: bool = False,
        ignored_parameters: Optional[Iterable[torch.nn.Parameter]] = None,
    ):
        """Updates FSDP's __init__ function to call _custom_auto_wrap.

        torch version must be 2.0.1.

        modified version of
        https://github.com/pytorch/pytorch/blob/96ca226a7332be0d8f3d6159d0c797e032ab0721/torch/distributed/fsdp/fully_sharded_data_parallel.py#L330
        """
        from torch.distributed.fsdp._dynamo_utils import _annotate_modules_for_dynamo
        from torch.distributed.fsdp._init_utils import (HYBRID_SHARDING_STRATEGIES, _check_orig_params_flattened,
                                                        _init_buffer_state, _init_core_state,
                                                        _init_ignored_module_states, _init_param_handle_from_module,
                                                        _init_prefetching_state, _init_process_group_state,
                                                        _init_runtime_state, _init_state_dict_state)
        from torch.distributed.fsdp._state_dict_utils import _register_all_state_dict_hooks
        from torch.distributed.fsdp._unshard_param_utils import _register_flat_param

        torch._C._log_api_usage_once('torch.distributed.fsdp')
        super(FullyShardedDataParallel, self).__init__()
        _init_ignored_module_states(self, module, ignored_modules, ignored_parameters)

        # Add module annotations for Dynamo support (see function for details)
        _annotate_modules_for_dynamo(module, self._ignored_modules, use_orig_params)

        # Initializes self.process_group, along with rank and world size. This will
        # also set another attribute, _inter_node_pg, to control the process group
        # over which sharding occurs, if sharding_strategy is {HYBRID_SHARD, _HYBRID_SHARD_ZERO2}.
        # Note that this is done before auto_wrapping, so that child FSDP modules simply pick up
        # the same process group state as the root FSDP module.
        _init_process_group_state(self, process_group, sharding_strategy, auto_wrap_policy)  # type: ignore
        if auto_wrap_policy is not None:
            auto_wrap_kwargs = {
                'module': module,
                'auto_wrap_policy': auto_wrap_policy,
                'wrapper_cls': FullyShardedDataParallel,
                'ignored_modules': self._ignored_modules,
                'ignored_params': self._ignored_params,
                'only_wrap_children': True,  # avoid double wrapping the root
            }
            fsdp_kwargs = {
                'process_group': process_group,
                'sharding_strategy': sharding_strategy,
                'cpu_offload': cpu_offload,
                'backward_prefetch': backward_prefetch,
                'mixed_precision': mixed_precision,
                'param_init_fn': param_init_fn,
                'device_id': device_id,
                'sync_module_states': sync_module_states,
                'forward_prefetch': forward_prefetch,
                'limit_all_gathers': limit_all_gathers,
                'use_orig_params': use_orig_params,
            }
            if sharding_strategy in HYBRID_SHARDING_STRATEGIES:
                # Share root process groups with children to maintain
                # the invariant that all FSDP modules will have the same
                # process groups.
                fsdp_kwargs['process_group'] = (self.process_group, self._inter_node_pg)

            # CHANGE: Call our custom _auto_wrap function
            _custom_auto_wrap_t2p0p1(auto_wrap_kwargs, fsdp_kwargs, FullyShardedDataParallel)

        backward_prefetch_limit = 1
        forward_prefetch_limit = 1
        _init_core_state(
            self,
            sharding_strategy,
            mixed_precision,
            cpu_offload,
            limit_all_gathers,
            use_orig_params,
            backward_prefetch_limit,
            forward_prefetch_limit,
        )
        _init_runtime_state(self)
        _init_prefetching_state(self, backward_prefetch, forward_prefetch)  # type: ignore
        _init_buffer_state(self, module)
        _init_param_handle_from_module(
            self,
            module,
            device_id,
            param_init_fn,
            sync_module_states,
            FullyShardedDataParallel,
        )
        self._fsdp_wrapped_module = module
        if not use_orig_params:
            _check_orig_params_flattened(self, self._ignored_params)
            _register_flat_param(self, self)

        # `_state_dict_type` controls the `state_dict()` behavior, which is
        # implemented using post-save and pre-load hooks
        _init_state_dict_state(self)
        _register_all_state_dict_hooks(self)


def get_split_size(dim_size: int, chunks: int) -> int:
    """Gets the minimum size per chunk.

    A tensor of dim_size 5 and 4 chunks will have chunks of size [2, 1, 1, 1].

    Args:
        dim_size(int): Size of the dimension being chunked.
        chunks(int): Number of chunks to create for ``dim_size``.

    Returns:
        An int indicating the split size to use.
    """
    return dim_size // chunks


def get_chunked_dim_size(dim_size: int, chunks: int, idx: int) -> int:
    """Computes the dim size of the chunk for provided ``idx`` given ``dim_size`` and ``chunks``.

    A tensor of dim_size 5 and 4 chunks will have chunks of size [2, 1, 1, 1].

    Args:
        dim_size(int): Size of the dimension being chunked.
        chunks(int): Number of chunks to create for ``dim_size``.
        idx(int): The index of chunk whose dim size is being requested.

    Returns:
        An int indicating the dim size of the chunk.
    """
    assert idx >= 0 and idx < chunks
    split_size = get_split_size(dim_size, chunks)
    if idx < dim_size % chunks:
        split_size += 1
    return split_size


def build_metadata(
    self: ChunkShardingSpec,
    tensor_sizes: torch.Size,
    tensor_properties: sharded_tensor_meta.TensorProperties,
) -> sharded_tensor_meta.ShardedTensorMetadata:
    """Updates ChunkShardingSpec's build_metadata fn to use a dynamic sharding dimension.

    modified version of
    https://github.com/pytorch/pytorch/blob/v2.0.1/torch/distributed/_shard/sharding_spec/chunk_sharding_spec.py#L77
    """
    tensor_num_dim = len(tensor_sizes)

    self._verify_dim(self.dim)
    if self.dim >= tensor_num_dim or self.dim < -tensor_num_dim:  # type: ignore[operator]
        raise ValueError(f'Invalid sharding dim: {self.dim}')

    shards_metadata = []
    while True:
        sharding_dim_size = tensor_sizes[self.dim]  # type: ignore[index]
        chunks = len(self.placements)

        if sharding_dim_size // chunks == 0:
            self.dim += 1  # type: ignore[operator]
        else:
            break
    current_offsets = [0] * tensor_num_dim
    for idx, placement in enumerate(self.placements):
        # generate ShardMetadata for each placement device
        chunked_dim_size = get_chunked_dim_size(sharding_dim_size, chunks, idx)
        if chunked_dim_size > 0:
            shard_size = list(tensor_sizes)
            shard_size[self.dim] = chunked_dim_size  # type: ignore[index]

            shard_metadata = ShardMetadata(
                shard_offsets=current_offsets.copy(),
                shard_sizes=shard_size,
                placement=placement,
            )
            shards_metadata.append(shard_metadata)

            current_offsets[self.dim] += chunked_dim_size  # type: ignore[index]
    self.dim = 0
    return sharded_tensor_meta.ShardedTensorMetadata(shards_metadata, tensor_sizes, tensor_properties)


def shard(self: ChunkShardingSpec, tensor: torch.Tensor, src_rank: int = 0, process_group=None) -> 'ShardedTensor':
    """Updates ChunkShardingSpec's shard fn to use a dynamic sharding dimension.

    modified version of
    https://github.com/pytorch/pytorch/blob/v2.0.1/torch/distributed/_shard/sharding_spec/chunk_sharding_spec.py#L116
    """
    # relative imports to avoid circular dependency
    from torch.distributed._shard.sharded_tensor import ShardedTensor

    tensor_properties = sharded_tensor_meta.TensorProperties(
        dtype=tensor.dtype,
        layout=tensor.layout,
        requires_grad=tensor.requires_grad,
        memory_format=torch.contiguous_format,
        pin_memory=tensor.is_pinned(),
    )
    current_rank = distributed.get_rank(process_group)
    tensor_meta = self.build_metadata(tensor.size(), tensor_properties)
    local_shards = []
    local_tensor = None
    local_metadata = None
    tensors_to_scatter = [None] * distributed.get_world_size(process_group)

    while True:
        sharding_dim_size = tensor.size()[self.dim]  # type: ignore[index]
        chunks = len(self.placements)

        if sharding_dim_size // chunks == 0:
            self.dim += 1  # type: ignore[operator]
        else:
            break
    scatter_shape = list(tensor.size())
    scatter_shape[self.dim] = get_chunked_dim_size(sharding_dim_size, chunks, 0)  # type: ignore[index]

    for shard_meta in tensor_meta.shards_metadata:
        rank, device = _parse_and_validate_remote_device(process_group, shard_meta.placement)
        if current_rank == src_rank:
            # Reshape to get shard for this rank and we don't want autograd
            # recording here for the narrow op and 'local_shard' should be a
            # leaf variable in the autograd graph.
            narrowed_tensor = narrow_tensor(tensor, shard_meta)
            if shard_meta.shard_sizes[self.dim] < scatter_shape[self.dim]:  # type: ignore[index]
                # for the last shard that might be smaller to other shards
                # resize the narrowed tensor to the same size and use it for
                # the scatter collective as dist.scatter requires same size
                # inputs on every rank
                tensor_to_scatter = (narrowed_tensor.detach().clone().resize_(scatter_shape))
            else:
                tensor_to_scatter = narrowed_tensor.detach().clone().contiguous()

            tensors_to_scatter[rank] = tensor_to_scatter  # type: ignore[index]

        if current_rank == rank:
            local_tensor = torch.empty(scatter_shape, dtype=tensor.dtype, layout=tensor.layout, device=device)
            local_metadata = shard_meta

    # each rank should have local_tensor and local_metadata initialized if we build
    # the metadata list in a correct way.
    assert local_tensor is not None
    assert local_metadata is not None

    # Scatter the shards to all ranks in the pg
    # scatter takes the global rank as ``src``
    src_for_scatter = src_rank
    if (process_group is not None and process_group is not distributed.distributed_c10d._get_default_group()):
        src_for_scatter = distributed.distributed_c10d.get_global_rank(process_group, src_for_scatter)

    distributed.scatter(
        local_tensor,
        scatter_list=tensors_to_scatter if current_rank == src_rank else None,
        src=src_for_scatter,
        group=process_group,
    )

    if list(local_tensor.size()) != local_metadata.shard_sizes:
        # detach again after receiving to ensure local shards remain a leaf node
        local_tensor = local_tensor.resize_(local_metadata.shard_sizes).detach()

    # Sync requires_grad to local_shard.
    local_tensor.requires_grad = tensor.requires_grad

    local_shards.append(Shard(tensor=local_tensor, metadata=local_metadata))

    st = ShardedTensor._init_from_local_shards_and_global_metadata(local_shards,
                                                                   tensor_meta,
                                                                   process_group=process_group)

    # Manually set sharding_spec
    st._sharding_spec = self
    self.dim = 0
    return st
