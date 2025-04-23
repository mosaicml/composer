# Copyright 2024 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

from typing import Callable, Optional

import pytest
import torch
from packaging import version

SKIP_TEST = version.parse(torch.__version__) < version.parse('2.6.0')
if not SKIP_TEST:
    # TODO (FSDP2) move this to top once we deprecate torch 2.5
    from composer.distributed import fsdp2
    prepare_fully_shard = fsdp2.prepare_fully_shard
    legalize_param_sharing_between_modules = fsdp2.legalize_param_sharing_between_modules
    get_standalone_and_tied_modules = fsdp2.get_standalone_and_tied_modules
else:
    prepare_fully_shard = lambda *args, **kwargs: None
    legalize_param_sharing_between_modules = lambda *args, **kwargs: None
    get_standalone_and_tied_modules = lambda *args, **kwargs: ([], set())

def fsdp2_context(func: Callable) -> Optional[Callable]:
    """Decorator to run tests with models initialized on the meta device for torch version 2.6+."""
    func = pytest.mark.skipif(SKIP_TEST, reason='Skipping test for torch version < 2.6.0')(func)
    func = pytest.mark.filterwarnings('ignore:FSDP2 Config/APIs are experimental*:UserWarning')(func)
    return func
