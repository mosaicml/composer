# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import contextlib
import datetime

import pytest

from composer.trainer.devices import DeviceGPU
from composer.utils import dist
from tests.common import world_size


@world_size(2)
@pytest.mark.parametrize('success', [True, False])
def test_run_local_rank_first_context(success):
    if success:
        dist.initialize_dist(DeviceGPU(), timeout=datetime.timedelta(seconds=30))
    with contextlib.nullcontext() if success else pytest.raises(RuntimeError):
        with dist.run_local_rank_zero_first():
            pass
