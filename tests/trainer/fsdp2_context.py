# Copyright 2024 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

from typing import Callable, Optional

import pytest
import torch
from packaging import version

SKIP_TEST = version.parse(torch.__version__) < version.parse('2.6.0')
if not SKIP_TEST:
    # TODO (FSDP2) move this to top once we decprecate torch 2.5
    from composer.distributed.fsdp2 import *


def fsdp2_context(func: Callable) -> Optional[Callable]:
    """Decorator to run tests with models initialized on the meta device for torch version 2.6+."""
    func = pytest.mark.skipif(SKIP_TEST, reason='Skipping test for torch version < 2.6.0')(func)
    func = pytest.mark.filterwarnings('ignore:FSDP2 Config/APIs are experimental*:UserWarning')(func)
    return func
