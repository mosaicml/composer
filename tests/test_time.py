# Copyright 2021 MosaicML. All Rights Reserved.

import contextlib
from typing import Dict, Tuple

import pytest

from composer.core.time import Time, Timer, TimeUnit

_INVALID_CONVERSIONS = [
    # from_unit,      to_unit,       drop_last
    (TimeUnit.EPOCH, TimeUnit.TOKEN, True),
    (TimeUnit.BATCH, TimeUnit.TOKEN, True),
    (TimeUnit.BATCH, TimeUnit.TOKEN, False),
    (TimeUnit.SAMPLE, TimeUnit.TOKEN, True),
    (TimeUnit.SAMPLE, TimeUnit.TOKEN, False),
    (TimeUnit.TOKEN, TimeUnit.BATCH, True),
    (TimeUnit.TOKEN, TimeUnit.BATCH, False),
    (TimeUnit.TOKEN, TimeUnit.SAMPLE, True),
    (TimeUnit.TOKEN, TimeUnit.SAMPLE, False),
    (TimeUnit.TOKEN, TimeUnit.EPOCH, True),
]


@pytest.mark.parametrize("from_unit", [TimeUnit.EPOCH, TimeUnit.BATCH, TimeUnit.SAMPLE, TimeUnit.TOKEN])
@pytest.mark.parametrize("to_unit", [TimeUnit.EPOCH, TimeUnit.BATCH, TimeUnit.SAMPLE, TimeUnit.TOKEN])
@pytest.mark.parametrize("drop_last", [True, False])
def test_time_convert(from_unit: TimeUnit, to_unit: TimeUnit, drop_last: bool):
    dataset_num_samples = 100
    batch_size = 30
    dataset_num_tokens = 504

    time_size = 200

    time = Time(time_size, from_unit)

    batches_per_epoch = dataset_num_samples // batch_size if drop_last else dataset_num_samples // batch_size + 1
    samples_per_epoch = batches_per_epoch * batch_size if drop_last else dataset_num_samples

    expected: Dict[Tuple[TimeUnit, TimeUnit, bool], int] = {
        # from unit,       to unit,      drop last
        (TimeUnit.EPOCH, TimeUnit.EPOCH, drop_last):
            time_size,
        (TimeUnit.EPOCH, TimeUnit.BATCH, drop_last):
            time_size * batches_per_epoch,
        (TimeUnit.EPOCH, TimeUnit.SAMPLE, drop_last):
            time_size * samples_per_epoch,
        (TimeUnit.EPOCH, TimeUnit.TOKEN, False):
            time_size * dataset_num_tokens,
        (TimeUnit.BATCH, TimeUnit.EPOCH, drop_last):
            time_size // batches_per_epoch,
        (TimeUnit.BATCH, TimeUnit.BATCH, drop_last):
            time_size,
        (TimeUnit.BATCH, TimeUnit.SAMPLE, True):
            time_size * batch_size,
        (TimeUnit.BATCH, TimeUnit.SAMPLE, False):
            (time_size // batches_per_epoch) * dataset_num_samples + (time_size % batches_per_epoch) * batch_size,
        (TimeUnit.SAMPLE, TimeUnit.EPOCH, drop_last):
            time_size // samples_per_epoch,
        (TimeUnit.SAMPLE, TimeUnit.BATCH, True):
            time_size // batch_size,
        (TimeUnit.SAMPLE, TimeUnit.BATCH, False):
            (time_size // samples_per_epoch) * batches_per_epoch + (time_size % samples_per_epoch) // batch_size,
        (TimeUnit.SAMPLE, TimeUnit.SAMPLE, drop_last):
            time_size,
        (TimeUnit.TOKEN, TimeUnit.EPOCH, False):
            time_size // dataset_num_tokens,
        (TimeUnit.TOKEN, TimeUnit.TOKEN, drop_last):
            time_size,
    }

    if (from_unit, to_unit, drop_last) in _INVALID_CONVERSIONS:
        ctx = pytest.raises(ValueError)
    else:
        ctx = contextlib.nullcontext()

    with ctx:
        conversion = time.convert(to_unit,
                                  dataset_num_samples=dataset_num_samples,
                                  dataset_num_tokens=dataset_num_tokens,
                                  batch_size=batch_size,
                                  drop_last=drop_last)
        assert conversion.unit == to_unit
        assert conversion.value == expected[from_unit, to_unit, drop_last]


def test_time_convert_to_duration():
    conversion = Time.from_timestring("200ep").convert("dur", max_training_duration="400ep")
    assert conversion == Time(0.5, TimeUnit.DURATION)


def test_time_convert_from_duration():
    conversion = Time.from_timestring("0.5dur").convert("ep", max_training_duration="400ep")
    assert conversion == Time(200, TimeUnit.EPOCH)


@pytest.mark.parametrize("time_string,expected_value,expected_unit", [
    ["1ep", 1, TimeUnit.EPOCH],
    ["2ba", 2, TimeUnit.BATCH],
    ["3e10sp", 3 * 10**10, TimeUnit.SAMPLE],
    ["4tok", 4, TimeUnit.TOKEN],
    ["0.5dur", 0.5, TimeUnit.DURATION],
])
def test_time_parse(time_string, expected_value, expected_unit):
    time = Time.from_timestring(time_string)
    assert time.value == expected_value
    assert time.unit == expected_unit


def test_time_math():
    t1 = Time.from_timestring("1ep")
    t2 = Time.from_timestring("2ep")
    t3 = Time.from_timestring("3ep")
    t4 = Time.from_timestring("0.5dur")
    assert t1 + t2 == t3
    assert t2 - t1 == t1
    assert t1 - t2 == -t1
    assert t1 < t2
    assert t1 <= t2
    assert t2 > t1
    assert t2 >= t1
    assert t3 >= t3
    assert t3 <= t3
    assert t4 * t2 == t1
    assert 0.5 * t2 == t1
    assert t4 * 2 == Time.from_timestring("1dur")
    assert t1 / t2 == t4
    assert t2 / 2 == t1


def test_time_repr():
    time = Time(1, "tok")
    assert repr(time) == "Time(1, TimeUnit.TOKEN)"
    assert eval(repr(time)) == time


def test_timer():
    timer = Timer()
    timer.on_batch_complete(10, 20)
    timer.on_epoch_complete()
    timer.on_batch_complete(5)
    assert timer.epoch.value == 1
    assert timer.batch.value == 2
    assert timer.sample.value == 15
    assert timer.token.value == 20
