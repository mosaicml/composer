# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Utilities for callbacks."""

import math
from typing import Callable, Union, Set, Optional

from composer.core import Event, State, Time, TimeUnit


def create_interval_scheduler(interval: Union[str, int, Time],
                              include_end_of_training: bool = True,
                              checkpoint_events: bool = True,
                              final_events: Optional[Set[Event]] = None) -> Callable[[State, Event], bool]:
    """Helper function to create a scheduler according to a specified interval.

    Args:
        interval (Union[str, int, :class:`.Time`]): If an integer, it will be assumed to be in :attr:`.TimeUnit.EPOCH`.
            Otherwise, the unit must be either :attr:`.TimeUnit.EPOCH`, :attr:`.TimeUnit.BATCH`,
            :attr:`.TimeUnit.TOKEN`, or :attr:`.TimeUnit.SAMPLE`.
        include_end_of_training (bool): If true, the returned callable will return true at the end of training as well.
            Otherwise, the returned callable will return true at intervals only.
        checkpoint_events (bool): If true, will use the EPOCH_CHECKPOINT and BATCH_CHECKPOINT events. If False, will use
            the EPOCH_END and BATCH_END events.
        final_events (Optional[Set[Event]]): The set of events to trigger on at the end of training.

    Returns:
        Callable[[State, Event], bool]: A function that returns true at interval and at the end of training if specified.
            For example, it can be passed as the ``save_interval`` argument into the :class:`.CheckpointSaver`.
    """
    if final_events is None:
        final_events = {Event.BATCH_CHECKPOINT, Event.EPOCH_CHECKPOINT}

    if isinstance(interval, str):
        interval = Time.from_timestring(interval)
    if isinstance(interval, int):
        interval = Time(interval, TimeUnit.EPOCH)

    if interval.unit == TimeUnit.EPOCH:
        interval_event = Event.EPOCH_CHECKPOINT if checkpoint_events else Event.EPOCH_END
    elif interval.unit in {TimeUnit.BATCH, TimeUnit.TOKEN, TimeUnit.SAMPLE, TimeUnit.DURATION}:
        interval_event = Event.BATCH_CHECKPOINT if checkpoint_events else Event.BATCH_END
    else:
        raise NotImplementedError(
            f'Unknown interval: {interval.unit}. Must be TimeUnit.EPOCH, TimeUnit.BATCH, TimeUnit.TOKEN, or TimeUnit.SAMPLE.'
        )

    last_batch_seen = -1

    def check_interval(state: State, event: Event):
        # `TimeUnit.Duration` value is a float from `[0.0, 1.0)`
        if not interval.unit == TimeUnit.DURATION and int(interval) <= 0:
            return False
        nonlocal last_batch_seen  # required to use the last_batch_seen from the outer function scope

        # Previous timestamp will only be None if training has not started, but we are returning False
        # in this case, just to be safe
        if state.previous_timestamp is None:
            return False

        elapsed_duration = state.get_elapsed_duration()
        assert elapsed_duration is not None, 'elapsed_duration is set on the BATCH_CHECKPOINT and EPOCH_CHECKPOINT'

        if include_end_of_training and event in final_events and elapsed_duration >= 1.0 and state.timestamp.batch != last_batch_seen:
            return True

        if interval.unit in {TimeUnit.EPOCH, TimeUnit.BATCH, TimeUnit.TOKEN, TimeUnit.SAMPLE}:
            previous_count = state.previous_timestamp.get(interval.unit)
            count = state.timestamp.get(interval.unit)
        # If the eval_interval is a duration, we will track progress in terms of the unit of max_duration
        elif interval.unit == TimeUnit.DURATION:
            assert state.max_duration is not None
            previous_count = state.previous_timestamp.get(state.max_duration.unit)
            count = state.timestamp.get(state.max_duration.unit)
        else:
            raise NotImplementedError(
                f'Unknown interval: {interval.unit}. Must be TimeUnit.EPOCH, TimeUnit.BATCH, TimeUnit.TOKEN, or TimeUnit.SAMPLE.'
            )

        threshold_passed = math.floor(previous_count / interval.value) != math.floor(count / interval.value)

        if interval.unit != TimeUnit.DURATION and event == interval_event and threshold_passed:
            last_batch_seen = state.timestamp.batch
            return True
        elif interval.unit == TimeUnit.DURATION:
            assert state.max_duration is not None, 'max_duration should not be None'
            if state.dataloader_len is None:
                raise RuntimeError(
                    f'Interval of type `dur` or {TimeUnit.DURATION} requires the dataloader to be sized.')
            
            if event == interval_event:
                if state.max_duration.unit == TimeUnit.EPOCH and int(
                        state.timestamp.batch) % math.ceil(state.max_duration.value * float(interval) *
                                                        state.dataloader_len) == 0:
                    last_batch_seen = state.timestamp.batch
                    return True
                elif state.max_duration.unit == TimeUnit.BATCH and int(state.timestamp.batch) % math.ceil(
                        state.max_duration.value * interval.value) == 0:
                    last_batch_seen = state.timestamp.batch
                    return True
                elif state.max_duration.unit == TimeUnit.SAMPLE:
                    samples_per_interval = math.ceil(state.max_duration.value * interval)
                    threshold_passed = math.floor(previous_count / samples_per_interval) != math.floor(
                        count / samples_per_interval)
                    if threshold_passed:
                        last_batch_seen = state.timestamp.batch
                        return True
                elif state.max_duration.unit == TimeUnit.TOKEN:
                    tokens_per_interval = math.ceil(state.max_duration.value * interval)
                    threshold_passed = math.floor(previous_count / tokens_per_interval) != math.floor(
                        count / tokens_per_interval)
                    if threshold_passed:
                        last_batch_seen = state.timestamp.batch
                        return True
        return False

    return check_interval
