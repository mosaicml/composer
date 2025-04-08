# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Distributed training."""

# TODO include fsdp2 after we deprecate torch-cpu 2.5
from composer.distributed.dist_strategy import (
    DDPSyncStrategy,
    ddp_sync_context,
    prepare_ddp_module,
    prepare_fsdp_module,
    prepare_tp_module,
)

__all__ = [
    'DDPSyncStrategy',
    'ddp_sync_context',
    'prepare_ddp_module',
    'prepare_fsdp_module',
    'prepare_tp_module',
]
