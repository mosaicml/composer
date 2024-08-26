# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Parallelism configs."""

from dataclasses import dataclass
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
