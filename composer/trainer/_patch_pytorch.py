# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

# Released under BSD 3-Clause License,
# Copyright (c) Facebook, Inc. and its affiliates.

# yapf: disable
# isort: skip_file
# pyright: reportGeneralTypeIssues=false

"""PyTorch, especially PyTorch Distributed, monkeypatches."""

import logging
from typing import no_type_check

import torch
from packaging import version

from composer.utils import dist

log = logging.getLogger(__name__)

def patch_unshard_for_automicrobatching(auto_microbatch_size_found=False):
    """Monkey patches sync hook into unshard when searching during automicrobatching."""
    from torch.distributed.fsdp._flat_param import FlatParamHandle
    if auto_microbatch_size_found:
        global original_unshard
        FlatParamHandle.unshard = (original_unshard)
    else:
        FlatParamHandle.unshard = (unshard_with_sync)

def patch_pytorch():
    """Monkey patches pytorch functions based on pytorch version."""
    if version.parse(torch.__version__) < version.parse('2.7.1'):
        # Monkey patch for torch < 2.7.1 ie torch == 2.7.0

        # No monkeypatches besides unshard (below)!
        pass


if version.parse(torch.__version__) >= version.parse('2.6.0') and version.parse(
        torch.__version__,
) < version.parse('2.7.1'):

    # Save original FlatParamHandle.unshard to revert back to when dropping automicrobatching hooks
    from torch.distributed.fsdp._flat_param import FlatParamHandle
    original_unshard = FlatParamHandle.unshard

    @no_type_check
    def unshard_with_sync(self):
        """Run the unshard logic, but with a sync after a :meth:`_alloc_padded_unsharded_flat_param`.

        This prevents deadlocks when some ranks OOM after the alloc call and others do not.
        This is a patched method from pytorch, meant to be called when automicrobatching
        turns on hooks in its search process for the optimal non-OOMing microbatch size.
        This includes all-gathering the flat parameter
        and switching to using the unsharded flat parameter. If the handle does
        not need unsharding, then this only switches to using the unsharded
        flat parameter. For ``NO_SHARD``, this is a no-op.
        If FSDP is in :meth:`summon_full_params` and the handle uses parameter
        mixed precision, then the parameter is forced to full precision.
        """
        if not self.needs_unshard():
            # Even when not needing an unshard, we should switch to using
            # the unsharded flat parameter
            unsharded_flat_param = (
                self._get_padded_unsharded_flat_param()
                if self.uses_sharded_strategy
                else self.flat_param
            )
            self._use_unsharded_flat_param(unsharded_flat_param)
            return
        unsharded_flat_param = self._alloc_padded_unsharded_flat_param()

        # Check if any other rank hit an OOM
        found_cuda_oom_tensor = torch.tensor([0], dtype=torch.uint8).to(self.device, non_blocking=True)

        dist.all_reduce(found_cuda_oom_tensor, reduce_operation='MAX')
        found_cuda_oom = found_cuda_oom_tensor.item()
        # Signal current rank is still in batch
        all_ranks_finished_tensor = torch.tensor([0], dtype=torch.uint8).to(self.device, non_blocking=True)

        dist.all_reduce(all_ranks_finished_tensor, reduce_operation='MIN')

        if found_cuda_oom == 1:
            raise RuntimeError('CUDA out of memory encountered on a different rank')
        padded_unsharded_flat_param = self._all_gather_flat_param(unsharded_flat_param)
        self._use_unsharded_flat_param(padded_unsharded_flat_param)
