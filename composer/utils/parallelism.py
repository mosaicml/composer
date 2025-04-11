# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Parallelism configs."""

import warnings
from dataclasses import dataclass, field
from typing import Any, Optional

from torch.distributed._tensor.device_mesh import DeviceMesh


@dataclass
class FSDPConfig:
    """Configuration for Fully Sharded Data Parallelism (FSDP)."""
    activation_checkpointing: bool = False
    activation_checkpointing_reentrant: bool = True
    activation_cpu_offload: bool = False
    auto_wrap: bool = True
    te_checkpoint_wrapper: bool = False
    te_shard_fp8_weight: bool = False
    backward_prefetch: str = 'BACKWARD_POST'
    backward_prefetch_limit: int = 1
    cpu_offload: bool = False
    data_parallel_shard_degree: int = -1
    data_parallel_replicate_degree: Optional[int] = None
    forward_prefetch: bool = False
    forward_prefetch_limit: int = 1
    ignored_modules: Optional[Any] = None
    keep_low_precision_grads: bool = False
    limit_all_gathers: bool = True
    load_monolith_rank0_only: bool = False
    load_planner: Optional[Any] = None
    mixed_precision: str = 'DEFAULT'
    process_group: Optional[Any] = None
    save_planner: Optional[Any] = None
    sharded_ckpt_prefix_dir: str = 'ep{epoch}-ba{batch}'
    sharding_strategy: str = 'FULL_SHARD'
    state_dict_type: str = 'full'
    sync_module_states: bool = False
    use_orig_params: bool = True
    verbose: bool = False

    _device_mesh: Optional[DeviceMesh] = field(default=None, init=False, repr=False)

    def __init__(self, **kwargs):
        if 'device_mesh' in kwargs or '_device_mesh' in kwargs:
            raise ValueError(
                f'Directly specifying device mesh for FSDP was deprecated in Composer version 0.24.0. ' +
                f"Please specify 'data_parallel_shard_degree' and/or 'data_parallel_replicate_degree' instead.",
            )

        for k, v in kwargs.items():
            setattr(self, k, v)

    @property
    def device_mesh(self) -> Optional[DeviceMesh]:
        return self._device_mesh

    @device_mesh.setter
    def device_mesh(self, value: Optional[DeviceMesh]):
        self._device_mesh = value


class _FSDP2ReadOnlyProperty:
    """Descriptor for FSDP2Config properties that are read-only. For FSDP1 compatibility/deprecation usage only.
    """
    
    def __init__(self, default_value):
        self.default_value = default_value
        
    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        return self.default_value
    
    def __set__(self, obj, value):
        raise AttributeError(f"Cannot modify read-only FSDP2Config attribute")


@dataclass
class FSDP2Config:
    """Configuration for Fully Sharded Data Parallelism (FSDP2).

    Args:
        device_mesh (Optional[DeviceMesh]): The DeviceMesh for sharding. If None, a default 1D mesh is created.
            For 1D mesh, parameters are fully sharded across the mesh (FSDP).
            For 2D mesh, parameters are sharded across the 1st dimension and replicated across the 0th dimension (HSDP).
        reshard_after_forward (Union[bool, int]): Controls parameter behavior after forward.
    """
    
    # Settable core FSDP2 parameters 
    device_mesh: Optional[DeviceMesh] = None
    reshard_after_forward: bool | int = True
    
    # Read-only compatibility properties
    # to be supported in FSDP2
    auto_wrap = _FSDP2ReadOnlyProperty(True)
    load_monolith_rank0_only = _FSDP2ReadOnlyProperty(False)
    sync_module_states = _FSDP2ReadOnlyProperty(False)
    activation_cpu_offload = _FSDP2ReadOnlyProperty(False)
    # to be deprecated in FSDP2
    state_dict_type = _FSDP2ReadOnlyProperty('sharded')
    use_orig_params = _FSDP2ReadOnlyProperty(True)

    
    def __post_init__(self):        
        warnings.warn('FSDP2 Config/APIs are experimental and subject to heavy changes', UserWarning)


@dataclass
class TPConfig:
    """Configuration for tensor parallelism (TP)."""
    device_mesh: Optional[DeviceMesh] = None
    tensor_parallel_degree: int = 1
    layer_plan: Any = None


@dataclass
class ParallelismConfig:
    """Configuration for parallelism."""
    fsdp: Optional[FSDPConfig] = None
    tp: Optional[TPConfig] = None
    fsdp2: Optional[FSDP2Config] = None
