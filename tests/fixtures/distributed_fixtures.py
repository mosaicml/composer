# Copyright 2021 MosaicML. All Rights Reserved.

import datetime

import pytest

from composer.utils import dist

DIST_TIMEOUT = datetime.timedelta(seconds=15)


@pytest.fixture(autouse=True)
def wait_for_all_procs():
    yield
    dist.barrier()
