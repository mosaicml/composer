# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import pytest

from composer.core import Event, Time, Timestamp
from composer.utils.misc import create_interval_scheduler, partial_format


class DummyState:

    def __init__(self, current_batches: int, max_duration: str, dataloader_len: str):
        self.previous_timestamp = Timestamp(batch=current_batches - 1)
        self.timestamp = Timestamp(batch=current_batches)
        self.max_duration = Time.from_timestring(max_duration)
        self.dataloader_len = Time.from_timestring(dataloader_len)

    def get_elapsed_duration(self):
        return 0


def test_partial_format():
    # No args provided
    assert partial_format('{foo} {bar} {}') == '{foo} {bar} {}'

    # Keyword args
    assert partial_format('{foo} {bar}', foo='Hello') == 'Hello {bar}'
    assert partial_format('{foo} {bar}', foo='Hello', bar='World') == 'Hello World'

    # Positional args
    assert partial_format('{} {}', 'Hello') == 'Hello {}'
    assert partial_format('{} {}', 'Hello', 'World') == 'Hello World'

    # Positional and keyword args
    assert partial_format('{foo} {}', 'World') == '{foo} World'
    assert partial_format('{foo} {}', foo='Hello') == 'Hello {}'
    assert partial_format('{foo} {}', 'World', foo='Hello') == 'Hello World'


@pytest.mark.parametrize(
    'interval,current_batches,max_duration,dataloader_len,expected',
    [
        ('0.25dur', 1, '1ep', '1ba', True),
        ('0.25dur', 1, '1ep', '4ba', True),
        ('0.25dur', 2, '1ep', '5ba', True),
        ('0.25dur', 1, '1ep', '5ba', False),
        ('0.25dur', 1, '1ba', '1ba', True),
        ('0.25dur', 1, '4ba', '4ba', True),
        ('0.25dur', 2, '5ba', '5ba', True),
        ('0.25dur', 1, '5ba', '5ba', False),
    ],
)
def test_interval_scheduler(
    interval: str,
    current_batches: int,
    max_duration: str,
    dataloader_len: str,
    expected: bool,
):
    interval_scheduler = create_interval_scheduler(interval)
    dummy_state = DummyState(current_batches, max_duration, dataloader_len)

    event = Event.BATCH_CHECKPOINT

    actual = interval_scheduler(dummy_state, event)  # type: ignore (intentional)
    assert actual == expected

class DummyStateWCT:

    def __init__(self, current_batches: int, max_duration: str, dataloader_len: str):
        self.previous_timestamp = Timestamp(total_wct=current_batches - 1)
        self.timestamp = Timestamp(total_wct=current_batches)
        self.max_duration = Time.from_timestring(max_duration)
        self.dataloader_len = Time.from_timestring(dataloader_len)

    def get_elapsed_duration(self):
        return 0

@pytest.mark.parametrize(
    'interval,current_batches,max_duration,dataloader_len,expected',
    [
        ('1sec', 1, '10sec', '10ba', True),
        ('2sec', 2, '20sec', '20ba', True),
        ('2sec', 1, '20sec', '20ba', False),
        ('10sec', 10, '100sec', '100ba', True),
        ('4sec', 4, '40sec', '40ba', True)
    ],
)
def test_interval_scheduler_seconds(
    interval: str,
    current_batches: int,
    max_duration: str,
    dataloader_len: str,
    expected: bool,
):
    interval_scheduler = create_interval_scheduler(interval)
    dummy_state = DummyStateWCT(current_batches, max_duration, dataloader_len)

    event = Event.BATCH_CHECKPOINT

    actual = interval_scheduler(dummy_state, event)
    assert actual == expected
