# Copyright 2021 MosaicML. All Rights Reserved.

import datetime
import pathlib

import pytest

from composer.utils import dist

DIST_TIMEOUT = datetime.timedelta(seconds=15)


@pytest.fixture(autouse=True)
def wait_for_all_procs():
    yield
    dist.barrier()


@pytest.fixture
def rank_zero_tmpdir(tmpdir: pathlib.Path):
    tmpdir_list = [str(tmpdir)]
    dist.broadcast_object_list(tmpdir_list, src=0)
    return pathlib.Path(tmpdir_list[0])
