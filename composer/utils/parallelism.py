# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Parallelism configs."""

import warnings
from dataclasses import dataclass
from typing import Any, Optional

from torch.distributed._tensor.device_mesh import DeviceMesh

from composer.utils.warnings import VersionedDeprecationWarning


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
    device_mesh: Optional[DeviceMesh] = None
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


def create_fsdp_config(fsdp_config: dict[str, Any]):
    """Modify fsdp_config to set default values for missing keys."""
    fsdp_config = {**fsdp_config}  # Shallow copy to avoid modifying input
    if 'process_group' in fsdp_config:
        warnings.warn(
            VersionedDeprecationWarning(
                'process_group is deprecated. Please specify `data_parallel_shard_degree` and `data_parallel_replicate_degree` instead.',
                remove_version='0.24',
            ),
        )

    if 'device_mesh' in fsdp_config:
        warnings.warn(
            VersionedDeprecationWarning(
                'device_mesh is deprecated. Please specify `data_parallel_shard_degree` and `data_parallel_replicate_degree` instead.',
                remove_version='0.24',
            ),
        )
        if 'data_parallel_shard_degree' in fsdp_config or 'data_parallel_replicate_degree' in fsdp_config:
            raise ValueError(
                'Cannot specify both `device_mesh` and `data_parallel_shard_degree` or `data_parallel_replicate_degree`. Please remove `device_mesh`.',
            )
        device_mesh = fsdp_config.pop('device_mesh')
        if len(device_mesh) == 1:
            fsdp_config['data_parallel_shard_degree'] = device_mesh[0]
        elif len(device_mesh) == 2:
            fsdp_config['data_parallel_replicate_degree'] = device_mesh[0]
            fsdp_config['data_parallel_shard_degree'] = device_mesh[1]
        else:
            raise ValueError(
                f'device_mesh must be of length 1 or 2 but received length {len(device_mesh)} with device mesh {device_mesh}.',
            )

    return FSDPConfig(**fsdp_config)


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
