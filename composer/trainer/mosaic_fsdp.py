# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

# Released under BSD 3-Clause License,
# Copyright (c) Facebook, Inc. and its affiliates.

"""Updates FSDPs _auto_wrap to enable module_kwargs and custom process_group cache."""

import functools
import warnings
from typing import Any, Callable, Dict, Set, Tuple, Union, cast

import torch
import torch.nn as nn
from torch import distributed
from torch.distributed import ProcessGroup
from torch.distributed.fsdp import (BackwardPrefetch, CPUOffload, FullyShardedDataParallel, MixedPrecision,
                                    ShardingStrategy)
from torch.distributed.fsdp._utils import _contains_batchnorm, _override_batchnorm_mixed_precision
from torch.distributed.fsdp.wrap import _or_policy, _wrap, _wrap_batchnorm_individually

from composer.core import Precision
from composer.utils import dist

__all__ = [
    'sharding_map',
    'backward_prefetch_map',
    'get_torch_dtype',
    'get_mixed_precision',
    'get_cpu_offload',
    'get_process_group',
    'MosaicFullyShardedDataParallel',
]

sharding_map = {
    'NO_SHARD': ShardingStrategy.NO_SHARD,
    'SHARD_GRAD_OP': ShardingStrategy.SHARD_GRAD_OP,
    'FULL_SHARD': ShardingStrategy.FULL_SHARD,
}

backward_prefetch_map = {
    'NONE': None,
    'BACKWARD_PRE': BackwardPrefetch.BACKWARD_PRE,
    'BACKWARD_POST': BackwardPrefetch.BACKWARD_POST,
}


def get_torch_dtype(dtype: Union[Precision, str]):
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

    mixed_precision = MixedPrecision(
        param_dtype=param_dtype,
        reduce_dtype=reduce_dtype,
        buffer_dtype=buffer_dtype,
        keep_low_precision_grads=keep_low_precision_grads,
    )

    return mixed_precision, param_dtype, reduce_dtype, buffer_dtype


def get_cpu_offload(cpu_offload=False):
    """Helper fn for configuring cpu_offload."""
    cpu_offload = CPUOffload(offload_params=True) if cpu_offload else None
    if cpu_offload is not None:
        raise ValueError('FSDP CPU Offload not supported yet.')
    return cpu_offload


def get_process_group(pg, process_group_cache=None):
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
            f'On rank={dist.get_global_rank()} using cached progress group with {ranks=}. ' +\
            'If the intention was to use a new process group, a new process group can be instantiated and passed' +\
            "in as an arguement (`'process_group': newly_instantiated_process_group_obect,`)"
        )
        return process_group_cache[ranks]

    warnings.warn(
        f'Composer is instantiating custom process groups with {ranks=} (on rank={dist.get_global_rank()}). ' +\
        'This is an experimental feature.'
    )

    ranks_per_subgroup_list = list(set(dist.all_gather_object(ranks)))
    current_group, _subgroups = distributed.distributed_c10d.new_subgroups_by_enumeration(ranks_per_subgroup_list)

    if process_group_cache is not None:
        process_group_cache[ranks] = current_group
    return current_group


def _custom_recursive_wrap(module: nn.Module,
                           auto_wrap_policy: Callable,
                           wrapper_cls: Callable,
                           ignored_modules: Set[nn.Module],
                           ignored_params: Set[nn.Parameter],
                           process_group_cache: Dict[Tuple[int], Any],
                           only_wrap_children: bool = False,
                           **kwargs: Any) -> Tuple[nn.Module, int]:
    """Updates FSDPs _recursive_wrap to enable module_kwargs and custom process_group cache.

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
            wrapped_child, num_wrapped_params = _custom_recursive_wrap(
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
            module_kwargs = module_kwargs if isinstance(module_kwargs, dict) else {}
            # backward_prefetch_map
            if 'sharding_strategy' in module_kwargs and module_kwargs['sharding_strategy'] not in sharding_map.values():
                module_kwargs['sharding_strategy'] = sharding_map[module_kwargs['sharding_strategy'].upper()]
            if 'backward_prefetch' in module_kwargs and module_kwargs[
                    'backward_prefetch'] not in backward_prefetch_map.values():
                module_kwargs['backward_prefetch'] = backward_prefetch_map[module_kwargs['backward_prefetch'].upper()]
            if 'cpu_offload' in module_kwargs and not isinstance(module_kwargs['cpu_offload'], CPUOffload):
                module_kwargs['cpu_offload'] = get_cpu_offload(cpu_offload=module_kwargs['cpu_offload'].upper())
            if 'mixed_precision' in module_kwargs and not isinstance(module_kwargs['mixed_precision'], MixedPrecision):
                # `precision` needs to set `'mixed_precision'`, but `precision` is not part of fsdp kwargs
                raise NotImplementedError(
                    f"Automated setting of custom per module mixed_precision is not implemented, but it can be set if `isinstance(module_kwargs['mixed_precision'], MixedPrecision)`"
                )
            if 'process_group' in module_kwargs:
                module_kwargs['process_group'] = get_process_group(module_kwargs['process_group'], process_group_cache)

            final_kwargs = {**kwargs, **module_kwargs}

            # Leaf node or final wrapping of the remainder both happen here.
            return _wrap(module, wrapper_cls, **final_kwargs), num_params
        else:
            return module, total_wrapped_params
    return module, 0


class MosaicFullyShardedDataParallel(FullyShardedDataParallel):
    """Updates FSDP's _auto_wrap to enable module_kwargs."""

    def _auto_wrap(
        self,
        auto_wrap_kwargs: Dict[str, Any],
        fsdp_kwargs: Dict[str, Any],
    ) -> None:
        """Updates _auto_wrap to enable module_kwargs.

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
        auto_wrap_kwargs['process_group_cache'] = {}
        _custom_recursive_wrap(**auto_wrap_kwargs, **fsdp_kwargs)
