# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Utilities for callbacks."""

import math
from typing import Callable, Union

from composer.core import Event, State, Time, TimeUnit


def create_interval_scheduler(interval: Union[str, int, Time],
                              include_end_of_training=True) -> Callable[[State, Event], bool]:
    """Helper function to create a scheduler according to a specified interval.

    Args:
        interval (Union[str, int, :class:`.Time`]): If an integer, it will be assumed to be in :attr:`.TimeUnit.EPOCH`.
            Otherwise, the unit must be either :attr:`.TimeUnit.EPOCH`, :attr:`.TimeUnit.BATCH`,
            :attr:`.TimeUnit.TOKEN`, or :attr:`.TimeUnit.SAMPLE`.
        include_end_of_training (bool): If true, the returned callable will return true at the end of training as well.
            Otherwise, the returned callable will return true at intervals only.

    Returns:
        Callable[[State, Event], bool]: A function that returns true at interval and at the end of training if specified.
            For example, it can be passed as the ``save_interval`` argument into the :class:`.CheckpointSaver`.
    """
    if isinstance(interval, str):
        interval = Time.from_timestring(interval)
    if isinstance(interval, int):
        interval = Time(interval, TimeUnit.EPOCH)

    if interval.unit == TimeUnit.EPOCH:
        save_event = Event.EPOCH_CHECKPOINT
    elif interval.unit in {TimeUnit.BATCH, TimeUnit.TOKEN, TimeUnit.SAMPLE}:
        save_event = Event.BATCH_CHECKPOINT
    else:
        raise NotImplementedError(
            f'Unknown interval: {interval.unit}. Must be TimeUnit.EPOCH, TimeUnit.BATCH, TimeUnit.TOKEN, or TimeUnit.SAMPLE.'
        )

    def check_interval(state: State, event: Event):
        elapsed_duration = state.get_elapsed_duration()
        assert elapsed_duration is not None, 'elapsed_duration is set on the BATCH_CHECKPOINT and EPOCH_CHECKPOINT'

        if include_end_of_training and elapsed_duration >= 1.0:
            return True

        # previous timestamp will only be None if training has not started, but we are returning False
        # in this case, just to be safe
        if state.previous_timestamp is None:
            return False

        if interval.unit in {TimeUnit.EPOCH, TimeUnit.BATCH, TimeUnit.TOKEN, TimeUnit.SAMPLE}:
            previous_count = state.previous_timestamp.get(interval.unit)
            count = state.timestamp.get(interval.unit)
        else:
            raise NotImplementedError(
                f'Unknown interval: {interval.unit}. Must be TimeUnit.EPOCH, TimeUnit.BATCH, TimeUnit.TOKEN, or TimeUnit.SAMPLE.'
            )

        threshold_passed = math.floor(previous_count / interval.value) != math.floor(count / interval.value)
        return event == save_event and threshold_passed

    return check_interval
