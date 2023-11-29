# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Miscellaneous Helpers."""

import math
import socket
from contextlib import contextmanager
from typing import TYPE_CHECKING, Callable, Optional, Set, Type, Union

import torch
from packaging import version
from torch.nn.parallel import DistributedDataParallel

if TYPE_CHECKING:
    from composer.core import Event, State, Time

__all__ = [
    'is_model_deepspeed',
    'is_model_fsdp',
    'is_notebook',
    'warning_on_one_line',
    'get_free_tcp_port',
    'model_eval_mode',
    'create_interval_scheduler',
]


def create_interval_scheduler(interval: Union[str, int, 'Time'],
                              include_end_of_training: bool = True,
                              checkpoint_events: bool = True,
                              final_events: Optional[Set['Event']] = None) -> Callable[['State', 'Event'], bool]:
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
    # inlined to avoid circular import
    from composer.core import Event, State, Time, TimeUnit

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
                if state.max_duration.unit == TimeUnit.EPOCH and int(state.timestamp.batch) % math.ceil(
                        state.max_duration.value * float(interval) * state.dataloader_len) == 0:
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


def is_model_deepspeed(model: torch.nn.Module) -> bool:
    """Whether ``model`` is an instance of a :class:`~deepspeed.DeepSpeedEngine`."""
    try:
        import deepspeed
    except ImportError:
        return False
    else:
        return isinstance(model, deepspeed.DeepSpeedEngine)


def is_model_ddp(model: torch.nn.Module) -> bool:
    """Whether ``model`` is an instance of a :class:`.DistributedDataParallel`."""
    return isinstance(model, DistributedDataParallel)


def is_model_fsdp(model: torch.nn.Module) -> bool:
    """Whether ``model`` is an instance of a :class:`.FullyShardedDataParallel`."""
    try:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

        if isinstance(model, FSDP):
            return True

        # Check if model is wrapped with FSDP
        for _, obj in model.named_children():
            if isinstance(obj, FSDP):
                return True
        return False
    except ImportError:
        return False


def is_notebook():
    """Whether Composer is running in a IPython/Jupyter Notebook."""
    try:
        __IPYTHON__  # type: ignore
        return True
    except NameError:
        return False


def warning_on_one_line(
    message: str,
    category: Type[Warning],
    filename: str,
    lineno: int,
    file=None,
    line=None,
):
    """Force Python warnings to consolidate into one line."""
    # From https://stackoverflow.com/questions/26430861/make-pythons-warnings-warn-not-mention-itself
    return f'{category.__name__}: {message} (source: {filename}:{lineno})\n'


def get_free_tcp_port() -> int:
    """Get free socket port to use as MASTER_PORT."""
    # from https://www.programcreek.com/python/?CodeExample=get+free+port
    tcp = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    tcp.bind(('', 0))
    _, port = tcp.getsockname()
    tcp.close()
    return port


@contextmanager
def model_eval_mode(model: torch.nn.Module):
    """Set model.eval() for context duration, restoring model status at end."""
    is_training = model.training
    try:
        model.eval()
        yield
    finally:
        model.train(mode=is_training)


def using_torch_2() -> bool:
    """Check the PyTorch version and compared it with version 2.0.0.

    Returns:
        bool: Return True if current version is greater than or equal to 2.0.0 else False
    """
    return version.parse(torch.__version__) >= version.parse('2.0.0')


def using_torch_2_0_1() -> bool:
    """Check the PyTorch version and compare it with version 2.0.1.

    Returns:
        bool: Return True if current version is greater than or equal to 2.0.1 else False
    """
    return version.parse(torch.__version__) >= version.parse('2.0.1')
