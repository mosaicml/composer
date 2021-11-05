# Copyright 2021 MosaicML. All Rights Reserved.

import os
import pathlib

import pytest
import torch.distributed


@pytest.fixture(autouse=True)
def ddp_setup(tmpdir: pathlib.Path) -> str:
    os.environ["DDP_TMPDIR"] = str(tmpdir / "store")
    yield
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
