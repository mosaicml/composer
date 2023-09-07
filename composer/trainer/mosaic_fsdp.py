# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

# Released under BSD 3-Clause License,
# Copyright (c) Facebook, Inc. and its affiliates.

"""Monkey patch FSDPs _auto_wrap to enable module_kwargs, custom process_group cache, and tensor sharding."""

import torch
from packaging import version
from torch.distributed.fsdp import FullyShardedDataParallel

from composer.trainer.mosaic_fsdp_utils import custom_auto_wrap_t1p13p1


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

    elif version.parse(torch.__version__) == version.parse('2.0.1'):
        # FullyShardedDataParallel monkey patch for torch == 2.0.1

        # Monkey patch __init__ where __init__ calls the custom _auto_wrap fn
        from composer.trainer.mosaic_fsdp_utils import init_fn_t2p0p1
        FullyShardedDataParallel.__init__ = init_fn_t2p0p1

    elif version.parse(torch.__version__) < version.parse('2.1.1'):
        # Monkey path for torch < 2.1.1 ie torch == 2.1.0

        # Monkey patch __init__ where __init__ calls the custom _auto_wrap fn
        from composer.trainer.mosaic_fsdp_utils import init_fn_t2p1p0
        FullyShardedDataParallel.__init__ = init_fn_t2p1p0

    elif version.parse(torch.__version__) >= version.parse('2.2.0'):
        raise NotImplementedError(
            f'FullyShardedDataParallel ui will be updated in torch2.1; _auto_wrap monkey patch needs to be updated accordingly.'
        )
