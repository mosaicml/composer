# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""FSDP related configs and helper functions."""

import logging
import warnings
from typing import Any, Union

import torch
from torch import distributed
from torch.distributed import ProcessGroup
from torch.distributed.fsdp import (
    BackwardPrefetch,
    CPUOffload,
    MixedPrecision,
    ShardingStrategy,
)

from composer.core import Precision
from composer.utils import dist

log = logging.getLogger(__name__)

SHARDING_MAP = {
    'NO_SHARD': ShardingStrategy.NO_SHARD,
    'SHARD_GRAD_OP': ShardingStrategy.SHARD_GRAD_OP,
    'FULL_SHARD': ShardingStrategy.FULL_SHARD,
    '_HYBRID_SHARD_ZERO2': ShardingStrategy._HYBRID_SHARD_ZERO2,
    'HYBRID_SHARD': ShardingStrategy.HYBRID_SHARD,
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
    if pg is None or isinstance(pg, ProcessGroup):  # Return as is, no caching
        return pg

    world_size = dist.get_world_size()
    local_world_size = dist.get_local_world_size()

    # Handle special str process_group cases
    if pg == 'self':
        pg = 'set1'
        log.info(f"Converting process_group='self' to process_group='{pg}'")
    elif pg == 'node':
        pg = f'set{local_world_size}'
        log.info(f"Converting process_group='node' to process_group='{pg}'")
    elif pg == 'local_rank_across_nodes':
        pg = f'mod{local_world_size}'
        log.info(f"Converting process_group='local_rank_across_nodes' to process_group='{pg}'")

    # Handle str and Union[list[int], tuple[int]] process_group cases
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
        log.info(f'Using cached progress group with {ranks=} on rank={dist.get_global_rank()}.')
        return process_group_cache[ranks]

    log.info(f'Instantiating custom process groups with {ranks=} on rank={dist.get_global_rank()}.')

    ranks_per_subgroup_list = list(set(dist.all_gather_object(ranks)))
    (
        current_group,
        _subgroups,
    ) = distributed.distributed_c10d.new_subgroups_by_enumeration(  # type: ignore[reportGeneralTypeIssues]
        ranks_per_subgroup_list,
    )

    if process_group_cache is not None:
        process_group_cache[ranks] = current_group
    return current_group


def set_custom_fsdp_module_kwargs(module_kwargs: dict, process_group_cache: dict[tuple[int], Any]) -> dict:
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
            f"Automated setting of custom per module mixed_precision is not implemented, but it can be set if `isinstance(module_kwargs['mixed_precision'], MixedPrecision)`",
        )
    if 'process_group' in module_kwargs:
        # Call on every process group if it is a tuple/list of non-ints
        if type(module_kwargs['process_group']) in [
            list,
            tuple,
        ] and not all(isinstance(x, int) for x in module_kwargs['process_group']):
            module_kwargs['process_group'] = tuple(
                _get_process_group(pg, process_group_cache) for pg in module_kwargs['process_group']
            )
        else:
            module_kwargs['process_group'] = _get_process_group(module_kwargs['process_group'], process_group_cache)

    return module_kwargs
