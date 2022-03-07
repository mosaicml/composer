# Copyright 2021 MosaicML. All Rights Reserved.

import pytest

from composer.core.time import Time, Timer, TimeUnit


@pytest.mark.parametrize("time_string,expected_value,expected_unit", [
    ["1ep", 1, TimeUnit.EPOCH],
    ["2ba", 2, TimeUnit.BATCH],
    ["3e10sp", 3 * 10**10, TimeUnit.SAMPLE],
    ["4tok", 4, TimeUnit.TOKEN],
    ["0.5dur", 0.5, TimeUnit.DURATION],
])
def test_time_parse(time_string: str, expected_value: int, expected_unit: TimeUnit):
    time = Time.from_timestring(time_string)
    assert time.value == expected_value
    assert time.unit == expected_unit


@pytest.mark.parametrize("expected_timestring,time", [
    ["1ep", Time(1, TimeUnit.EPOCH)],
    ["2ba", Time(2, TimeUnit.BATCH)],
    ["3sp", Time(3, TimeUnit.SAMPLE)],
    ["4tok", Time(4, TimeUnit.TOKEN)],
    ["0.5dur", Time(0.5, TimeUnit.DURATION)],
])
def test_to_timestring(expected_timestring: str, time: Time):
    assert time.to_timestring() == expected_timestring


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
    assert timer.epoch == 1
    assert timer.batch == 2
    assert timer.batch_in_epoch == 1
    assert timer.sample == 15
    assert timer.sample_in_epoch == 5
    assert timer.token == 20
    assert timer.token_in_epoch == 0
