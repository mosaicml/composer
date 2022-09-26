# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import contextlib
import datetime

import pytest

from composer.trainer.devices import DeviceGPU
from composer.utils import dist
from tests.common import world_size


@world_size(2)
def test_run_local_rank_first_context(world_size):
    with pytest.raises(RuntimeError) as e:
        with dist.run_local_rank_zero_first():
            pass
    assert 'If calling this function outside Trainer' in str(e) # Verify error raised is intended
