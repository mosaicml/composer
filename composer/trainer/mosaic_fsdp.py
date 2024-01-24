# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

# Released under BSD 3-Clause License,
# Copyright (c) Facebook, Inc. and its affiliates.

"""Monkey patch FSDPs _auto_wrap to enable module_kwargs and custom process_group cache and ChunkShardingSpec to enable sharding over all gpus."""

# pyright: reportGeneralTypeIssues=false
import torch
from packaging import version
from torch.distributed._shard.sharding_spec import ChunkShardingSpec
from torch.distributed.fsdp import FullyShardedDataParallel

from composer.trainer.mosaic_fsdp_utils import (_sharded_pre_load_state_dict_hook, build_metadata,
                                                custom_auto_wrap_t1p13p1)


def patch_pytorch():
    """Monkey patches pytorch functions based on pytorch version."""
    if version.parse(torch.__version__) < version.parse('1.13.1'):
        raise NotImplementedError(f'Not supported for torch < 1.13.1')

    elif version.parse(torch.__version__) < version.parse('2.0.0'):
        # Monkey patch for torch < 2.0 ie torch == 1.13.1

        # Monkey patch _auto_wrap with _custom_auto_wrap fn
        FullyShardedDataParallel._auto_wrap = custom_auto_wrap_t1p13p1  # type: ignore

    elif version.parse(torch.__version__) < version.parse('2.0.1'):
        raise NotImplementedError(f'Not supported for torch == 2.0.0')

    elif version.parse(torch.__version__) < version.parse('2.0.2'):
        # Monkey patch for torch == 2.0.1

        # Monkey patch __init__ where __init__ calls the custom _auto_wrap fn
        from composer.trainer.mosaic_fsdp_utils import init_fn_t2p0p1

        FullyShardedDataParallel.__init__ = init_fn_t2p0p1  # type: ignore

        # Monkey patch sharding method
        ChunkShardingSpec.build_metadata = build_metadata

    elif version.parse(torch.__version__) < version.parse('2.1.1'):
        # Monkey patch for torch < 2.1.1 ie torch == 2.1.0

        # Monkey patch sharding method
        ChunkShardingSpec.build_metadata = build_metadata

        # Monkey patch partial state dict handling
        from torch.distributed.fsdp import _state_dict_utils
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

    elif version.parse(torch.__version__) < version.parse('2.3.1'):
        # Monkey patch for torch < 2.3.1 ie torch == 2.3.0
        # Note: this is the same patch as 2.2.0, we are just making a new if branch
        # for clarity and modularity of changes.

        # Allow 2D HSDP
        from torch.distributed.fsdp import _runtime_utils
        _runtime_utils._validate_and_get_hybrid_shard_state = lambda *args, **kwargs: None

        # Monkeypath state_dict
        from composer.trainer.mosaic_fsdp_utils import init_fn_t2p2p0
        FullyShardedDataParallel.__init__ = init_fn_t2p2p0

        # Monkeypath state_dict
        from torch.distributed.checkpoint import state_dict  # type: ignore

        from composer.trainer.mosaic_fsdp_utils import _verify_options_t2p2p0
        state_dict._verify_options = _verify_options_t2p2p0

        # Monkeypatch sharding optim state
        from torch.distributed.fsdp import _optim_utils

        from composer.trainer.mosaic_fsdp_utils import _shard_orig_param_state
        _optim_utils._shard_orig_param_state = _shard_orig_param_state
