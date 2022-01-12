# Copyright 2021 MosaicML. All Rights Reserved.

import contextlib
from typing import Dict, Tuple

import pytest

from composer.core.time import Time, TimeUnit
from composer.utils._time_conversion import convert

_INVALID_CONVERSIONS = [
    # from_unit,      to_unit,       drop_last
    (TimeUnit.BATCH, TimeUnit.TOKEN, True),
    (TimeUnit.BATCH, TimeUnit.TOKEN, False),
    (TimeUnit.SAMPLE, TimeUnit.TOKEN, True),
    (TimeUnit.SAMPLE, TimeUnit.TOKEN, False),
    (TimeUnit.TOKEN, TimeUnit.BATCH, True),
    (TimeUnit.TOKEN, TimeUnit.BATCH, False),
    (TimeUnit.TOKEN, TimeUnit.SAMPLE, True),
    (TimeUnit.TOKEN, TimeUnit.SAMPLE, False),
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
        (TimeUnit.EPOCH, TimeUnit.TOKEN, drop_last):
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
        (TimeUnit.TOKEN, TimeUnit.EPOCH, drop_last):
            time_size // dataset_num_tokens,
        (TimeUnit.TOKEN, TimeUnit.TOKEN, drop_last):
            time_size,
    }

    if (from_unit, to_unit, drop_last) in _INVALID_CONVERSIONS:
        ctx = pytest.raises(ValueError)
    else:
        ctx = contextlib.nullcontext()

    with ctx:
        conversion = convert(time,
                             to_unit,
                             samples_per_epoch=samples_per_epoch,
                             dataset_num_tokens=dataset_num_tokens,
                             steps_per_epoch=batches_per_epoch)
        assert conversion.unit == to_unit
        assert conversion.value == expected[from_unit, to_unit, drop_last]


def test_time_convert_to_duration():
    conversion = convert("200ep", "dur", max_training_duration="400ep")
    assert conversion == Time(0.5, TimeUnit.DURATION)


def test_time_convert_from_duration():
    conversion = convert("0.5dur", "ep", max_training_duration="400ep")
    assert conversion == Time(200, TimeUnit.EPOCH)
