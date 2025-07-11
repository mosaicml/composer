# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Parallelism configs."""

import warnings
from dataclasses import dataclass, field, fields
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


@dataclass
class FSDP2Config:
    """Configuration for Fully Sharded Data Parallelism (FSDP2).

    Args:
        device_mesh (Optional[DeviceMesh]): The DeviceMesh for sharding. If None, a default 1D mesh is created.
            For 1D mesh, parameters are fully sharded across the mesh (FSDP).
            For 2D mesh, parameters are sharded across the 1st dimension and replicated across the 0th dimension (HSDP).
        reshard_after_forward (Union[bool, int]): Controls parameter behavior after forward.
        activation_checkpointing (bool): Whether to use activation checkpointing. Defaults to False.
        activation_cpu_offload (bool): Whether to use activation CPU offloading. Defaults to False.
        state_dict_type (str): Type of state dict to use. Can be 'full' or 'sharded'. Defaults to 'sharded'.
            - Note: In cases where `load_path` is not set in Trainer, `state_dict_type` indicates how a model will be saved.
            - Note: In cases where `load_path` is set in Trainer, `state_dict_type` indicates how a model will be loaded and also saved.
        load_monolith_rank0_only (bool): Whether to load monolithic checkpoints on rank 0 only. Defaults to False.
            - Note: when `load_monolith_rank0_only` is True and `load_path` is set in `Trainer`, `state_dict_type` must be 'full'.
        mixed_precision (str): Mixed precision to use. Can be 'DEFAULT', 'PURE', or 'FULL'. Defaults to 'DEFAULT'.
        verbose (bool): Whether to print verbose output. Defaults to False.
    """

    # Settable core FSDP2 attrs
    device_mesh: Optional[DeviceMesh] = None
    reshard_after_forward: bool | int = True
    # TODO: If we have reasonable evidence that activation checkpointing/activation offloading is decoupled from FSDP(2)
    #       in most of our use cases, we can decouple these two attributes from the FSDP2Config class.
    activation_checkpointing: bool = False
    activation_cpu_offload: bool = False
    state_dict_type: str = 'sharded'
    load_monolith_rank0_only: bool = False
    mixed_precision: str = 'DEFAULT'

    verbose: bool = False

    # Settable attrs that are automatically set during training
    _sync_module_states: bool = field(default=False, init=False, repr=False)

    @property
    def sync_module_states(self) -> bool:
        return self._sync_module_states

    @sync_module_states.setter
    def sync_module_states(self, value: bool):
        self._sync_module_states = value

    @classmethod
    def settable_attrs(cls) -> set[str]:
        """Return a set of all settable attributes of FSDP2Config."""
        return {field.name for field in fields(cls) if not field.name.startswith('_')}

    @classmethod
    def from_compatible_attrs(cls, attrs: dict[str, Any]) -> 'FSDP2Config':
        """Create an FSDP2Config by filtering FSDP2 compatible attributes from given attrs.

        Only attributes that are valid for FSDP2Config will be used, and warnings will be issued
        for any attributes that cannot be transferred. Therefore it supports both FSDP1 and FSDP2 attributes, and main
        use case is FSDP1 backwards compatibility.

        Args:
            attrs (dict[str, Any]): Dictionary of FSDP1/2 configuration attributes.

        Returns:
            FSDP2Config: A new FSDP2Config instance with compatible attributes.

        Warnings:
            UserWarning: If an attribute in the input dictionary is not a settable attribute
                         of FSDP2Config and will be ignored.
        """
        # Get the settable attributes of FSDP2Config
        settable_attrs = cls.settable_attrs()
        # Filter the input attributes to only include settable ones
        valid_attrs = {}
        for key, value in attrs.items():
            if key in settable_attrs:
                valid_attrs[key] = value
            else:
                warnings.warn(
                    f"Attribute '{key}: {value}' is not a settable attribute of FSDP2Config and will be ignored",
                    UserWarning,
                )

        # Create and return a new FSDP2Config with the valid attributes
        return FSDP2Config(**valid_attrs)

    ### Read-only properties for FSDP 1 compatibility ###
    @property
    def load_planner(self) -> Optional[Any]:
        return None

    @property
    def save_planner(self) -> Optional[Any]:
        return None

    @property
    def sharded_ckpt_prefix_dir(self) -> str:
        return 'ep{epoch}-ba{batch}'

    @property
    def data_parallel_shard_degree(self) -> int:
        return -1

    @property
    def data_parallel_replicate_degree(self) -> Optional[int]:
        return None

    @property
    def use_orig_params(self) -> bool:
        return True

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
