# Copyright 2021 MosaicML. All Rights Reserved.

import pytest
import torch.distributed


@pytest.fixture(autouse=True)
def ddp_cleanup():
    yield
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
