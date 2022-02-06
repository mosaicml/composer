"""
These fixtures are shared globally across the test suite
"""

from unittest.mock import Mock

import pytest

from composer.core import Logger, State
from composer.core.types import Precision


@pytest.fixture
def minimal_state():
    """Most minimally defined state possible. Tests should
    configure the state for their specific needs.
    """
    return State(model=Mock(),
                 precision=Precision.FP32,
                 grad_accum=1,
                 train_dataloader=Mock(__len__=lambda x: 100),
                 evaluators=Mock(),
                 max_duration='100ep')


@pytest.fixture
def empty_logger(minimal_state: State) -> Logger:
    """Logger without any output configured
    """
    return Logger(state=minimal_state, backends=[])
