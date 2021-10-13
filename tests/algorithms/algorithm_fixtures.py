# Copyright 2021 MosaicML. All Rights Reserved.

import pytest

from composer.core.state import State
from composer.loggers import Logger


@pytest.fixture()
def noop_dummy_logger(dummy_state: State) -> Logger:
    return Logger(state=dummy_state, backends=[])
