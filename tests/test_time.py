# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import pytest

from composer.core.time import Time, Timestamp, TimeUnit


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


def test_timestamp():
    timestamp = Timestamp()
    time = Time(10, "ep")
    assert timestamp < time
    assert timestamp.get(time.unit) == Time.from_epoch(0)


def test_timestamp_update():
    timestamp = Timestamp(epoch=1)
    timestamp_2 = timestamp.copy(batch=2)
    assert timestamp_2.epoch == 1
    assert timestamp_2.batch == 2
    assert timestamp_2.sample == 0
    assert timestamp is not timestamp_2


def test_timestamp_to_next_batch_epoch():
    timestamp = Timestamp()
    timestamp = timestamp.to_next_batch(10, 20)
    assert timestamp.batch == 1
    assert timestamp.batch_in_epoch == 1
    assert timestamp.batch_in_epoch == 1
    assert timestamp.sample == 10
    assert timestamp.sample_in_epoch == 10
    assert timestamp.token == 20
    assert timestamp.token_in_epoch == 20
    timestamp = timestamp.to_next_epoch()
    timestamp = timestamp.to_next_batch(5)
    assert timestamp.epoch == 1
    assert timestamp.batch == 2
    assert timestamp.batch_in_epoch == 1
    assert timestamp.sample == 15
    assert timestamp.sample_in_epoch == 5
    assert timestamp.token == 20
    assert timestamp.token_in_epoch == 0


def test_timestamp_repr():
    timestamp = Timestamp()
    assert timestamp == eval(repr(timestamp))
