# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

# Released under BSD 3-Clause License,
# Copyright (c) Facebook, Inc. and its affiliates.

# yapf: disable
# isort: skip_file

"""Utilities for monkey patching FSDP."""

import functools
import logging
import math
import warnings
import contextlib
from dataclasses import asdict
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, List, Optional, Set, Tuple, Union, cast, no_type_check

import torch
import torch.distributed._shard.sharded_tensor.metadata as sharded_tensor_meta
import torch.nn as nn
import torch.nn.functional as F
from packaging import version
from torch import distributed
from torch.distributed import ProcessGroup
from torch.distributed._shard.sharding_spec import ShardMetadata
from torch.distributed._shard.sharding_spec._internals import get_chunked_dim_size, get_split_size
from torch.distributed.distributed_c10d import get_process_group_ranks
from torch.distributed.fsdp import (
    BackwardPrefetch, CPUOffload, FullyShardedDataParallel, MixedPrecision,
    ShardingStrategy,
)
from torch.distributed.fsdp._fsdp_extensions import _ext_pre_load_state_dict_transform
from torch.distributed.utils import _replace_by_prefix

from composer.core import Precision
from composer.utils import dist

if TYPE_CHECKING:
    if version.parse(torch.__version__) >= version.parse('2.0.1') and version.parse(
            torch.__version__,
    ) < version.parse('2.2.0'):
        from torch.distributed.fsdp._common_utils import _FSDPState


log = logging.getLogger(__name__)

SHARDING_MAP = {
    'NO_SHARD': ShardingStrategy.NO_SHARD,
    'SHARD_GRAD_OP': ShardingStrategy.SHARD_GRAD_OP,
    'FULL_SHARD': ShardingStrategy.FULL_SHARD,
}

if version.parse(torch.__version__) >= version.parse('2.1.0'):
    SHARDING_MAP['_HYBRID_SHARD_ZERO2'] = ShardingStrategy._HYBRID_SHARD_ZERO2
    SHARDING_MAP['HYBRID_SHARD'] = ShardingStrategy.HYBRID_SHARD

BACKWARD_PREFETCH_MAP = {
    'NONE': None,
    'BACKWARD_PRE': BackwardPrefetch.BACKWARD_PRE,
    'BACKWARD_POST': BackwardPrefetch.BACKWARD_POST,
}

logger = logging.getLogger(__name__)


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
        log.info(f'Using cached progress group with {ranks=} on rank={dist.get_global_rank()}.')
        return process_group_cache[ranks]

    log.info(f'Instantiating custom process groups with {ranks=} on rank={dist.get_global_rank()}.')

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
            f"Automated setting of custom per module mixed_precision is not implemented, but it can be set if `isinstance(module_kwargs['mixed_precision'], MixedPrecision)`",
        )
    if 'process_group' in module_kwargs:
        # Call on every process group if it is a tuple/list of non-ints
        if type(module_kwargs['process_group']) in [
                list, tuple,
        ] and not all(isinstance(x, int) for x in module_kwargs['process_group']):
            module_kwargs['process_group'] = tuple(
                _get_process_group(pg, process_group_cache) for pg in module_kwargs['process_group']
            )
        else:
            module_kwargs['process_group'] = _get_process_group(module_kwargs['process_group'], process_group_cache)

    return module_kwargs

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
    This modification enables the user to pass custom FSDP arguments for every wrapped module.
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
                        f'FSDP with custom process groups cannot use `use_orig_params: True` when using meta init.',
                    )

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
    This modification enables the user to pass custom FSDP arguments for every wrapped module.
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
            raise ValueError(
                f'Expected {module_name} to NOT be FullyShardedDataParallel '
                'if using an `auto_wrap_policy`',
            )
    mixed_precision = fsdp_kwargs['mixed_precision']
    if mixed_precision is not None and _contains_batchnorm(root_module):
        _override_batchnorm_mixed_precision(root_module)
        auto_wrap_policy = functools.partial(_or_policy, policies=[_wrap_batchnorm_individually, auto_wrap_policy])
        warnings.warn(
            'Both mixed precision and an `auto_wrap_policy` were specified '
            'for FSDP, where the wrapped module has batch norm submodules. '
            'The batch norm submodules will be wrapped as separate FSDP '
            'instances with mixed precision disabled since some batch norm '
            'kernels do not support low precision.',
        )
    auto_wrap_kwargs['auto_wrap_policy'] = auto_wrap_policy

    # CHANGE: Add process group cache and call our custom _recursive_wrap
    auto_wrap_kwargs['process_group_cache'] = {}
    _custom_recursive_wrap_t2p0p1(**auto_wrap_kwargs, **fsdp_kwargs)


def build_metadata(
    self,
    tensor_sizes: torch.Size,
    tensor_properties: sharded_tensor_meta.TensorProperties,
) -> sharded_tensor_meta.ShardedTensorMetadata:
    """Adds nightly change for ChunkShardingSpec.

    Change implemented in https://github.com/pytorch/pytorch/pull/108915
    """
    tensor_num_dim = len(tensor_sizes)

    self._verify_dim(self.dim)
    if self.dim >= tensor_num_dim or self.dim < -tensor_num_dim:  # type: ignore[operator]
        raise ValueError(f'Invalid sharding dim: {self.dim}')

    shards_metadata = []
    sharding_dim_size = tensor_sizes[self.dim]  # type: ignore[index]
    chunks = len(self.placements)
    split_size = get_split_size(sharding_dim_size, chunks)
    for idx, placement in enumerate(self.placements):
        # generate ShardMetadata for each placement device
        chunked_dim_size = get_chunked_dim_size(sharding_dim_size, split_size, idx)
        shard_size = list(tensor_sizes)
        current_offsets = [0] * tensor_num_dim
        current_offsets[self.dim] = split_size * idx  # type: ignore[index]
        shard_size[self.dim] = chunked_dim_size  # type: ignore[index]

        shard_metadata = ShardMetadata(
            shard_offsets=current_offsets,
            shard_sizes=shard_size,
            placement=placement,
        )
        shards_metadata.append(shard_metadata)

    return sharded_tensor_meta.ShardedTensorMetadata(shards_metadata, tensor_sizes, tensor_properties)


@no_type_check
def _sharded_pre_load_state_dict_hook(
    module: nn.Module,
    fsdp_state: '_FSDPState',
    state_dict: Dict[str, Any],
    prefix: str,
) -> None:
    """Adds nightly change for partial state dict error handling.

    https://github.com/pytorch/pytorch/blob/0511df0ee9edeb5c2613805ccfb49beb323b87f9/torch/distributed/fsdp/_state_dict_utils.py#L607-L615

    The hook combines the unflattened, sharded parameters (ShardedTensor) to
    a new FlatParameter and shards the new FlatParameter to the local chunk.
    """
    from torch.distributed._tensor import Replicate
    from torch.distributed.distributed_c10d import _get_pg_default_device
    from torch.distributed.fsdp._common_utils import FSDP_PREFIX, _has_fsdp_params, _is_composable, _module_handle
    from torch.distributed.fsdp._runtime_utils import _lazy_init
    from torch.distributed.fsdp._state_dict_utils import _enter_unshard_params_ctx, _param_name_infos

    _lazy_init(fsdp_state, module)
    if not _is_composable(fsdp_state):
        _replace_by_prefix(state_dict, prefix, prefix + f'{FSDP_PREFIX}')
    if not _has_fsdp_params(fsdp_state, module):
        return

    handle = _module_handle(fsdp_state, module)
    if not handle.uses_sharded_strategy:
        raise RuntimeError(
            'load_sharded_state_dict can only be called when parameters '
            'are flattened and sharded.',
        )

    device = fsdp_state.compute_device
    for fqn, _, _ in _param_name_infos(module, fsdp_state):
        if not _is_composable(fsdp_state):
            fqn_from_global_root = f'{prefix}{FSDP_PREFIX}{fqn}'
        else:
            fqn_from_global_root = f'{prefix}{fqn}'
        try:
            param = state_dict.pop(fqn_from_global_root)
        except KeyError:
            logger.warning(
                f'Did not find param with FQN {fqn_from_global_root}, skipping it. '  # noqa: G004
                'The weight will not be filled if you expect it to be.',
            )
            continue  # TODO: Improve unittesting for state_dict finetuning
            # cases: https://github.com/pytorch/pytorch/issues/109134

        if not fsdp_state._state_dict_config.use_dtensor:
            # All-gather the param (ShardedTensor)
            param, shards = _ext_pre_load_state_dict_transform(param)

            assert len(shards) < 2, (
                'Expects 0 or 1 shard per rank '
                f'but got {len(shards)} shards on rank {fsdp_state.rank}.'
            )
            param_numel = param.size().numel()
            dim_0_size = param.size()[0]
            chunk_size = (math.ceil(dim_0_size / fsdp_state.world_size) * param_numel // dim_0_size)
            if len(shards) == 1:
                local_tensor = shards[0].tensor.flatten()
                pg_device = _get_pg_default_device(fsdp_state.process_group)
                if local_tensor.device.type != pg_device.type:
                    local_tensor = local_tensor.to(pg_device)
                num_padding = chunk_size - local_tensor.numel()
                if num_padding > 0:
                    local_tensor = F.pad(local_tensor, [0, num_padding])
            else:
                local_tensor = torch.zeros(chunk_size, dtype=param.dtype, device=device)
            tensor = torch.empty(
                chunk_size * fsdp_state.world_size,
                dtype=local_tensor.dtype,
                device=device,
            )
            if local_tensor.is_cpu:
                # Tensor could be on FSDP GPU compute device, while local_tensor is on CPU.
                # Convert to CPU so all_gather can work.
                tensor_dev = tensor.device
                tensor = tensor.cpu()
                tensor_list = list(torch.chunk(tensor, torch.distributed.get_world_size(fsdp_state.process_group)))
                torch.distributed.all_gather(tensor_list, local_tensor, group=fsdp_state.process_group)
                tensor.to(tensor_dev)
            else:
                torch.distributed.all_gather_into_tensor(tensor, local_tensor, group=fsdp_state.process_group)
            tensor = tensor.narrow(0, 0, param_numel).reshape(param.size())
            state_dict[fqn_from_global_root] = tensor
        else:
            if param.device != fsdp_state._device_mesh.device_type:
                param = param.to(fsdp_state._device_mesh.device_type)

            param = param.redistribute(device_mesh=param.device_mesh, placements=[Replicate()])
            state_dict[fqn_from_global_root] = param.to_local()

    _enter_unshard_params_ctx(module, fsdp_state, writeback=True)


if version.parse(torch.__version__) >= version.parse('2.2.1') and version.parse(
        torch.__version__,) < version.parse('2.2.3'):

    from torch.distributed.fsdp._optim_utils import FSDPParamInfo
    from torch.distributed.checkpoint._state_dict_utils import _gather_state_dict

    @no_type_check
    def _shard_orig_param_state(
        fsdp_param_info: FSDPParamInfo,
        fqn: str,
        optim_state: Dict[str, Any],
    ) -> Dict[str, Any]:
        if not optim_state:
            return {}
        fsdp_state = fsdp_param_info.state
        flat_param = fsdp_param_info.handle.flat_param
        param_idx = fsdp_param_info.param_indices[fqn]
        shard_param_info = flat_param._shard_param_infos[param_idx]  # type: ignore[attr-defined]
        optim_state = _gather_state_dict(
            optim_state, pg=fsdp_state.process_group, device=fsdp_state.compute_device,
        )
        if not shard_param_info.in_shard:
            return {}
        # Flatten and shard the state.
        new_optim_state: Dict[str, Any] = {}
        intra_param_start_idx = shard_param_info.intra_param_start_idx
        intra_param_end_idx = shard_param_info.intra_param_end_idx
        for state_name, value in optim_state.items():
            if (
                torch.is_tensor(value)
                and value.dim() > 0
                and fsdp_state.sharding_strategy != ShardingStrategy.NO_SHARD
            ):
                # This clone() is the patch to fix the OOM
                # https://github.com/pytorch/pytorch/pull/117261
                value = value.flatten()[intra_param_start_idx : intra_param_end_idx + 1].clone()  # type: ignore[operator]
            new_optim_state[state_name] = value
        return new_optim_state


if version.parse(torch.__version__) >= version.parse('2.3.0') and version.parse(
        torch.__version__,
) < version.parse('2.3.1'):
    from torch.distributed._tensor import DTensor

    @no_type_check
    def _same_storage(a, b):
        if isinstance(a, DTensor):
            a = a._local_tensor
        if isinstance(b, DTensor):
            b = b._local_tensor
        return a.untyped_storage().data_ptr() == b.untyped_storage().data_ptr()

    from torch.distributed.checkpoint.state_dict import (_unflatten_model_state_dict, _verify_options,
                                                         _load_model_state_dict, gc_context,
                                                         _verify_state_dict, _load_optim_state_dict,
                                                         FQNS_T)

    @no_type_check
    def _get_fqns(
        model: nn.Module,
        name: str,
        skip_ddp_prefix: bool = True,
        skip_compiler_prefix: bool = True,
    ) -> FQNS_T:
        """Used to convert the name of a parameter to the FQNs.

        For FSDP without `use_orig_params`, the name of FlatParameter can be mapped to
        multiple original parameters. As a result, the return type of this function
        is `Set[str]`.

        Args:
            module (nn.Module): the root model.
            name (str): the name
            skip_ddp_prefix (bool): whether to skip DDP's `module` prefix

        Returns:
            The canonical FQNs based on the model traversal.
        """
        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import _CHECKPOINT_PREFIX
        from torch.nn.parallel import DistributedDataParallel as DDP
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.checkpoint.state_dict import FLAT_PARAM
        from torch.distributed.fsdp._common_utils import FSDP_WRAPPED_MODULE

        # Remove the checkpoint prefix, if it exists.
        name = name.replace(_CHECKPOINT_PREFIX, '')
        if '.' not in name:
            return {name}

        obj_names = name.split('.')
        fqn_obj_names = []
        curr_obj = model
        for i, curr_obj_name in enumerate(obj_names):
            if isinstance(curr_obj, DDP):
                assert curr_obj_name == 'module'
                curr_obj = curr_obj.module
                if not skip_ddp_prefix:
                    fqn_obj_names.append(curr_obj_name)
            elif isinstance(curr_obj, FSDP):
                if i < len(obj_names) - 1 and obj_names[i + 1] == FLAT_PARAM:
                    prefix = '.'.join(fqn_obj_names)
                    flat_param = getattr(curr_obj, FLAT_PARAM)
                    if prefix:
                        prefix = f'{prefix}.'
                    return {f'{prefix}{fqn}' for fqn in flat_param._fqns}
                curr_obj = getattr(curr_obj, FSDP_WRAPPED_MODULE)
                if curr_obj_name != FSDP_WRAPPED_MODULE:
                    fqn_obj_names.append(curr_obj_name)
                    curr_obj = getattr(curr_obj, curr_obj_name)
            elif isinstance(curr_obj, torch._dynamo.eval_frame.OptimizedModule):
                assert curr_obj_name == '_orig_mod'
                curr_obj = curr_obj._orig_mod
                if not skip_compiler_prefix:
                    fqn_obj_names.append(curr_obj_name)
            else:
                fqn_obj_names.append(curr_obj_name)
                curr_obj = getattr(curr_obj, curr_obj_name)

        return {'.'.join(fqn_obj_names).replace(_CHECKPOINT_PREFIX, '')}

    def set_model_state_dict(
        model: nn.Module,
        model_state_dict,
        *,
        options = None,
    ):
        """Load the model state_dict.

        The counterpart of ``get_model_state_dict`` to set the state_dict to the
        model. See ``set_state_dict`` for the detail usage.

        Args:
            model (nn.Module): the nn.Module to the model.
            model_state_dict: (Dict[str, ValueType]):
            the model state_dict to load. If the key of the ``model_state_dict``
            is nn.Module, the key is a submodule of ``model`` and the value should
            be the state_dict of the submodule. When loading the state_dict,
            the prefix of the submodule will be append to the state_dict.
            options (StateDictOptions): the options to control how
                model state_dict and optimizer state_dict should be loaded. See
                `StateDictOptions` for the details.

        Returns:
            ``NamedTuple`` with ``missing_keys`` and ``unexpected_keys`` fields:
                * **missing_keys** is a list of str containing the missing keys
                * **unexpected_keys** is a list of str containing the unexpected keys

        :type model_state_dict: typing.Dict[str, ValueType]
        """
        from torch.distributed.fsdp._runtime_utils import _lazy_init
        for module in model.modules():
            if isinstance(module, FullyShardedDataParallel):
                _lazy_init(module, module)
        model_state_dict = _unflatten_model_state_dict(
            model, model_state_dict,
        )
        with gc_context():
            info = _verify_options(model, tuple(), optim_only=False, options=options)

            _verify_state_dict(model_state_dict, {}, info)
            return _load_model_state_dict(model, model_state_dict, info)

    def set_optimizer_state_dict(
        model: nn.Module,
        optimizers: Union[torch.optim.Optimizer, Iterable[torch.optim.Optimizer]],
        *,
        optim_state_dict,
        options = None,
    ) -> None:
        """Load the optimizers state_dict.

        The counterpart of ``get_optimizer_state_dict`` to set the state_dict to the
        optimizers. See ``set_state_dict`` for the detail usage.

        Args:
            model (nn.Module): the nn.Module to the model.
            optimizers (Union[Optimizer, Iterable[Optimizer]]):
                The optimizers that are used to optimize ``model``.
            optim_state_dict: OptimizerStateType:
                the optimizer state_dict to load.
            options (StateDictOptions): the options to control how
                model state_dict and optimizer state_dict should be loaded. See
                `StateDictOptions` for the details.

        Returns:
            None

        :type optim_state_dict: typing.OptimizerStateType
        """
        from torch.distributed.fsdp._runtime_utils import _lazy_init
        for module in model.modules():
            if isinstance(module, FullyShardedDataParallel):
                _lazy_init(module, module)
        with gc_context():
            optimizers = (
                (optimizers,)
                if isinstance(optimizers, torch.optim.Optimizer)
                else tuple(optimizers)
            )
            info = _verify_options(model, optimizers, optim_only=True, options=options)

            _verify_state_dict({}, optim_state_dict, info)
            _load_optim_state_dict(model, optimizers, optim_state_dict, info)

    ## Monkeypatches for multiple unshard streams

    from torch.distributed.utils import _p_assert, _to_kwargs
    from torch.distributed.distributed_c10d import get_process_group_ranks
    from torch.distributed.fsdp._common_utils import _is_composable, _FSDPState
    from torch.distributed.fsdp._runtime_utils import _root_cast_forward_input, _cast_buffers_to_dtype_and_device, _get_buffers_and_dtypes_for_computation, _reset_flat_param_grad_info_if_needed, HOMOGENEOUS_ATTR_NAMES, _pre_forward, _pre_forward_unshard, _post_forward, _post_forward_reshard, _lazy_init


    def _fsdp_state_has_default_pg(state: _FSDPState) -> bool:
        if state.process_group is None:
            # If no process group is attached to the _FSDPState, assume it uses default process group.
            return True
        return len(get_process_group_ranks(state.process_group)) == dist.get_world_size()


    def _fsdp_state_pg_ranks(state: _FSDPState) -> Tuple[int, ...]:
        if state.process_group is None:
            return tuple(range(dist.get_world_size()))
        else:
            return tuple(get_process_group_ranks(state.process_group))


    def _wait_for_computation_stream(
        computation_stream: torch.Stream,
        unshard_streams: Set[torch.Stream],
        pre_unshard_stream: torch.Stream,
    ):
        """_wait_for_computation_stream monkeypatch.

        Has the unshard and pre-unshard streams wait for the computation stream.
        For example, this should be called in the FSDP root's pre-forward to
        respect optimizer step computation.
        """
        # Tracing does not need to wait
        if torch.distributed._functional_collectives.is_torchdynamo_compiling():
            return
        ## MONKEYPATCH START
        for unshard_stream in unshard_streams:
            unshard_stream.wait_stream(computation_stream)
        ## MONKEYPATCH END
        # Having the pre-all-gather stream wait for the current stream even if we
        # do not leverage the pre-all-gather stream is tolerable since this only
        # runs once per iteration
        pre_unshard_stream.wait_stream(computation_stream)  # type: ignore[attr-defined]


    @no_type_check
    def _root_pre_forward(
        state: _FSDPState,
        module: nn.Module,
        args,
        kwargs,
    ) -> None:
        """_root_pre_forward monkeypatch.

        Runs pre-forward logic specific to the root FSDP instance, which should run
        before any individual module's pre-forward. This starts with an attempt at
        lazy initialization (which only runs non-vacuously once). Otherwise, if
        this is called on a non-root FSDP instance, then it returns directly.

        Args:
            state (_FSDPState): State of the FSDP instance.
            module (nn.Module): Module for which this logic tries to run. It may or
                may not be the root. If not, then this method does not do anything.
            args (Tuple): Arguments to the forward method of the module.
            kwargs (Dict): Keyword arguments to the forward method of the module.
        """
        with torch.profiler.record_function('FullyShardedDataParallel._root_pre_forward'):
            _lazy_init(state, module)
            _p_assert(state._is_root is not None, 'Expects a root FSDP to have been set')
            if not state._is_root:
                # Always cast forward inputs in the root of this local FSDP unit for mixed
                # precision, as this is where mixed precision could be configed.
                # This is more useful for auto wrapping that is recommended in composable path.
                # For manual wrapping, cast forward inputs on each local FSDP unit root will
                # increase some overhead, so not turned on for model wrapper path right now where
                # manual wrapping is more broadly used.
                if _is_composable(state):
                    return _root_cast_forward_input(state, module, args, kwargs)
                return args, kwargs

            # We cast buffers back to full precision if we're forcing full precision. Disjointly, we check if buffers
            # are in full precision and if we should cast them back to lower precision, which happens when
            # exiting eval() mode.
            handle = state._handle
            if handle:
                should_cast_buffers_to_full_prec = handle._force_full_precision
            else:
                should_cast_buffers_to_full_prec = True

            if should_cast_buffers_to_full_prec:
                _cast_buffers_to_dtype_and_device(
                    buffers=dict(module.named_buffers()).values(),
                    buffer_dtypes=list(state._buffer_name_to_orig_dtype.values()),
                    device=state.compute_device,
                )
                # This flag is only set when we cast buffers to full precision, to avoid the
                # CPU overhead that can stem from retrieving all buffers and their types in the
                # following else branch.
                state._needs_buffer_dtype_restore_check = True
            elif getattr(state, '_needs_buffer_dtype_restore_check', False):
                # Check if buffers are in full precision and we need to cast them
                # back down.
                (
                    buffers,
                    buffer_dtypes_for_computation,
                ) = _get_buffers_and_dtypes_for_computation(state, module)
                if len(buffers) > 0 and len(buffer_dtypes_for_computation) > 0:
                    if any(
                        buffer.dtype != buffer_dtype_for_computation
                        for buffer, buffer_dtype_for_computation in zip(
                            buffers, buffer_dtypes_for_computation,
                        )
                    ):
                        # Assume we have to cast everything if there is one mismatch
                        _cast_buffers_to_dtype_and_device(
                            buffers, buffer_dtypes_for_computation, state.compute_device,
                        )
                # We don't have to check this again until we cast buffers to full precision again.
                state._needs_buffer_dtype_restore_check = False

            if state.forward_prefetch:
                handles = []
                for fsdp_state in state._all_fsdp_states:
                    if fsdp_state._handle:
                        handles.append(fsdp_state._handle)
                for handle in handles:
                    handle._needs_pre_forward_unshard = True
                    handle._prefetched = False
            _wait_for_computation_stream(
                state._device_handle.current_stream(),
                state._all_unshard_streams,  # MONKEYPATCH
                state._pre_unshard_stream,
            )
            _reset_flat_param_grad_info_if_needed(state._all_handles)

            # Prepares the forward inputs by moving them to ``compute_device``
            # TODO: Do not use the side stream for tensor copies for now; investigate
            # the perf with/without it.
            with torch.profiler.record_function('FullyShardedDataParallel._to_kwargs'):
                args_tuple, kwargs_tuple = _to_kwargs(
                    args, kwargs, state.compute_device, False,
                )
            args = args_tuple[0]
            kwargs = kwargs_tuple[0]

            return _root_cast_forward_input(state, module, args, kwargs)


    @no_type_check
    def _share_state_and_init_handle_attrs(
        root_state: _FSDPState,
        root_module: nn.Module,
    ) -> None:
        """_share_state_and_init_handle_attrs monkeypatch.

        Shares data structure state from the ``root_state`` to all FSDP states in
        ``root_module`` 's module tree, and initializes handle attributes. These
        are done together to require a single loop over the states.
        """
        handle = root_state._handle
        if handle:
            handle.init_flat_param_attributes()
        attr_name_to_values: Dict[str, Set[Any]] = {}
        for attr_name in HOMOGENEOUS_ATTR_NAMES:
            attr_name_to_values[attr_name] = set()
        root_state._all_handles = root_state._exec_order_data.all_handles  # share reference
        # Update _has_optim_in_backward for each handle.
        for handle in root_state._all_handles:
            flat_param = handle.flat_param
            if hasattr(flat_param, '_in_backward_optimizers'):
                raise RuntimeError(
                    'FSDP optimizer in backward only supported with use_orig_params=True!',
                )
            handle._has_optim_in_backward = flat_param._params is not None and any(
                hasattr(param, '_in_backward_optimizers') for param in flat_param._params
            )
            if handle._has_optim_in_backward:
                torch._C._log_api_usage_once('fsdp.optimizer_in_backward')
        ## MONKEYPATCH START
        # Keep track of any new unshard streams we may have to add for specific process groups.
        fsdp_pg_unshard_streams = {}
        ## MONKEYPATCH END
        for fsdp_state in root_state._all_fsdp_states:
            for attr_name in HOMOGENEOUS_ATTR_NAMES:
                _p_assert(
                    hasattr(fsdp_state, attr_name),
                    f'FSDP state missing attribute {attr_name}',
                )
                attr_name_to_values[attr_name].add(getattr(fsdp_state, attr_name))
            if fsdp_state is root_state:
                root_state_pg_ranks = tuple(range(dist.get_world_size()))
                fsdp_pg_unshard_streams[root_state_pg_ranks] = root_state._unshard_stream
                continue
            # Relax the assert for non-root FSDP instances in case the nested
            # initialized module is wrapped again in FSDP later (e.g. after
            # training to run inference)
            _p_assert(
                fsdp_state._is_root is None or not fsdp_state._is_root,
                "Non-root FSDP instance's `_is_root` should not have been "
                'set yet or should have been set to `False`',
            )
            fsdp_state._is_root = False

            ## MONKEYPATCH START
            # Take care of any new unshard streams we have to create for non-default process groups.
            if _fsdp_state_has_default_pg(fsdp_state):
                # If using default process group, unshard stream is the same as root fsdp instance.
                fsdp_state._unshard_stream = root_state._unshard_stream
            else:
                # Otherwise, unshard stream is separate.
                state_pg_ranks = _fsdp_state_pg_ranks(fsdp_state)
                print(f'Process group ranks: {state_pg_ranks} in {fsdp_pg_unshard_streams.keys()} is {state_pg_ranks in fsdp_pg_unshard_streams}')
                if state_pg_ranks in fsdp_pg_unshard_streams:
                    # Reuse already created the unshard stream for this process group.
                    fsdp_state._unshard_stream = fsdp_pg_unshard_streams[state_pg_ranks]
                else:
                    # Create new unshard stream for this process group.
                    fsdp_state._unshard_stream = fsdp_state._device_handle.Stream()
                    fsdp_pg_unshard_streams[state_pg_ranks] = fsdp_state._unshard_stream
                    print(f'\tNew stream {fsdp_state._unshard_stream} for process group {state_pg_ranks}')
            ## MONKEYPATCH END

            # All other stream assignments stay common across all of FSDP.
            fsdp_state._post_backward_stream = root_state._post_backward_stream
            fsdp_state._pre_unshard_stream = root_state._pre_unshard_stream
            fsdp_state._all_reduce_stream = root_state._all_reduce_stream
            fsdp_state._default_stream = root_state._default_stream
            fsdp_state._exec_order_data = root_state._exec_order_data
            fsdp_state._free_event_queue = root_state._free_event_queue
            if fsdp_state._fsdp_extension is not None:
                fsdp_state._fsdp_extension.compute_stream = root_state._default_stream
            handle = fsdp_state._handle
            if handle:
                handle.init_flat_param_attributes()
        ## MONKEYPATCH START
        root_state._all_unshard_streams = set(fsdp_pg_unshard_streams.values())
        ## MONKEYPATCH END
        for attr_name, attr_values in attr_name_to_values.items():
            if len(attr_values) != 1:
                raise ValueError(
                    f'Expects one homogeneous value for {attr_name} but got {attr_values}',
                )

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Run the forward pass for the wrapped module, inserting FSDP-specific pre- and post-forward sharding logic."""
        handle = self._handle
        with torch.autograd.profiler.record_function(
            "FullyShardedDataParallel.forward"
        ):
            args, kwargs = _root_pre_forward(self, self, args, kwargs)
            unused = None
            args, kwargs = _pre_forward(
                self,
                handle,
                _pre_forward_unshard,
                self._fsdp_wrapped_module,
                args,
                kwargs,
            )
            if handle:
                _p_assert(
                    handle.flat_param.device == self.compute_device,
                    "Expected `FlatParameter` to be on the compute device "
                    f"{self.compute_device} but got {handle.flat_param.device}",
                )
            output = self._fsdp_wrapped_module(*args, **kwargs)
            return _post_forward(
                self, handle, _post_forward_reshard, self, unused, output
            )
    
    # DEBUG MONKEYPATCHES
        
    from torch.distributed.fsdp._runtime_utils import _prefetch_handle, _unshard, _PrefetchMode
        
    @no_type_check
    def _pre_forward_unshard(
        state: _FSDPState,
        handle,
    ) -> None:
        """Unshards parameters in the pre-forward."""
        if not handle:
            return
        # If the handles have been prefetched, then there is no need to call
        # `_unshard()` again
        if not handle._prefetched:
            _unshard(state, handle, state._unshard_stream, state._pre_unshard_stream)
        handle._needs_pre_forward_unshard = False
        # Don't wait during trace
        print(f'_pre_forward_unshard {state._device_handle.current_stream()} wait on stream {state._unshard_stream}')
        # print(f'\t_pre_forward_unshard wait on stream {tuple(get_process_group_ranks(state._unshard_stream))}')
        if not torch.distributed._functional_collectives.is_torchdynamo_compiling():
            state._device_handle.current_stream().wait_stream(state._unshard_stream)
        with torch.profiler.record_function(
            "FullyShardedDataParallel._pre_forward_prefetch"
        ):
            _prefetch_handle(state, handle, _PrefetchMode.FORWARD)
