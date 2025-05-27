# Copyright 2024 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

from typing import Callable, Optional

import pytest

from composer.distributed import activation_checkpointing, fsdp2, fsdp2_utils, prepare_distributed
apply_ac = activation_checkpointing.apply_ac
parallelize_model = prepare_distributed.parallelize_model
legalize_param_sharing_between_modules = fsdp2_utils.legalize_param_sharing_between_modules
get_standalone_and_tied_modules = fsdp2_utils.get_standalone_and_tied_modules
_recursive_apply_fully_shard = fsdp2._recursive_apply_fully_shard
_generate_default_policy = fsdp2_utils.generate_default_policy
check_param_tying = fsdp2_utils.check_param_tying


def fsdp2_context(func: Callable) -> Optional[Callable]:
    """Decorator to run tests with models initialized on the meta device for torch version 2.6+."""
    func = pytest.mark.filterwarnings('ignore:FSDP2 Config/APIs are experimental*:UserWarning')(func)
    return func
