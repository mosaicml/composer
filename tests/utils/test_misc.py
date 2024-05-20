import datetime
import pytest

from composer.core import Event, Time, Timestamp
from composer.utils.misc import create_interval_scheduler, partial_format

class DummyState:
    def __init__(self, current_batches: int, max_duration: str, dataloader_len: str, seconds_per_batch: int):
        self.previous_timestamp = Timestamp(batch=current_batches - 1, total_wct=datetime.timedelta(seconds=(current_batches-1)* seconds_per_batch))
        self.timestamp = Timestamp(batch=current_batches - 1, total_wct=datetime.timedelta(seconds=current_batches* seconds_per_batch))
        self.max_duration = Time.from_timestring(max_duration)
        self.dataloader_len = Time.from_timestring(dataloader_len)
        self.seconds_per_batch = seconds_per_batch
        self.total_elapsed_time = datetime.timedelta(seconds=current_batches * seconds_per_batch)

    def get_elapsed_duration(self):
        return self.total_elapsed_time.total_seconds() / self.max_duration.value

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
    'interval,current_batches,max_duration,dataloader_len,seconds_per_batch,expected',
    [
        ('0.25dur', 1, '1ep', '1ba', 10, True),
        ('0.25dur', 1, '1ep', '4ba', 10, True),
        ('0.25dur', 2, '1ep', '5ba', 10, True),
        ('0.25dur', 1, '1ep', '5ba', 10, True),
        ('0.25dur', 1, '1ba', '1ba', 10, True),
        ('0.25dur', 1, '4ba', '4ba', 10, True),
        ('0.25dur', 2, '5ba', '5ba', 10, True),
        ('0.25dur', 1, '5ba', '5ba', 10, True),
        ('10sec', 1, '6ba', '1ba', 10, True),
        ('10sec', 5, '6ba', '1ba', 10, True),
        ('10sec', 6, '6ba', '1ba', 10, True),
        ('20sec', 2, '6ba', '1ba', 1, False),
    ],
)
def test_interval_scheduler(
    interval: str,
    current_batches: int,
    max_duration: str,
    dataloader_len: str,
    seconds_per_batch: int,
    expected: bool,
):
    interval_scheduler = create_interval_scheduler(interval)
    dummy_state = DummyState(current_batches, max_duration, dataloader_len, seconds_per_batch)

    event = Event.BATCH_CHECKPOINT

    actual = interval_scheduler(dummy_state, event)
    assert actual == expected

