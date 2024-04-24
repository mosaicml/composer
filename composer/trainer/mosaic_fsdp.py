# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

# Released under BSD 3-Clause License,
# Copyright (c) Facebook, Inc. and its affiliates.

"""Monkey patch FSDPs _auto_wrap to enable module_kwargs and custom process_group cache and ChunkShardingSpec to enable sharding over all gpus."""

# pyright: reportGeneralTypeIssues=false
import torch
from packaging import version
from torch.distributed._shard.sharding_spec import ChunkShardingSpec


def patch_pytorch():
    """Monkey patches pytorch functions based on pytorch version."""
    if version.parse(torch.__version__) < version.parse('2.1.1'):
        # Monkey patch for torch < 2.1.1 ie torch == 2.1.0

        # Monkey patch sharding method
        from composer.trainer.mosaic_fsdp_utils import build_metadata

        ChunkShardingSpec.build_metadata = build_metadata

        # Monkey patch partial state dict handling
        from torch.distributed.fsdp import _state_dict_utils

        from composer.trainer.mosaic_fsdp_utils import _sharded_pre_load_state_dict_hook

        _state_dict_utils._sharded_pre_load_state_dict_hook = (_sharded_pre_load_state_dict_hook)

        # Allow 2D HSDP
        from torch.distributed.fsdp import _runtime_utils
        _runtime_utils._validate_and_get_hybrid_shard_state = lambda *args, **kwargs: None

    elif version.parse(torch.__version__) < version.parse('2.1.3'):
        # Monkey patch for torch < 2.1.3 ie torch == 2.1.1, 2.1.2

        # Allow 2D HSDP
        from torch.distributed.fsdp import _runtime_utils
        _runtime_utils._validate_and_get_hybrid_shard_state = lambda *args, **kwargs: None

    elif version.parse(torch.__version__) < version.parse('2.2.1'):
        # Monkey patch for torch < 2.2.1 ie torch == 2.2.0

        # Allow 2D HSDP
        from torch.distributed.fsdp import _runtime_utils
        _runtime_utils._validate_and_get_hybrid_shard_state = lambda *args, **kwargs: None

    elif version.parse(torch.__version__) < version.parse('2.2.3'):
        # Monkey patch for torch < 2.2.3 ie torch == 2.2.1/2.2.2 currently

        # Fix memory leak for FSDP.optim_state_dict_to_load
        # https://github.com/pytorch/pytorch/issues/116553
        from torch.distributed.fsdp import _optim_utils

        from composer.trainer.mosaic_fsdp_utils import _shard_orig_param_state
        _optim_utils._shard_orig_param_state = _shard_orig_param_state

    elif version.parse(torch.__version__) < version.parse('2.3.1'):
        # Monkey patch for torch < 2.3.1 ie torch == 2.3.0

        # Monkeypatch _flat_param.py to fix 2D with SHARD_GRAD_OP
        # Issue: https://github.com/pytorch/pytorch/issues/123272
        from torch.distributed.fsdp import _flat_param

        from composer.trainer.mosaic_fsdp_utils import _same_storage
        _flat_param._same_storage = _same_storage
