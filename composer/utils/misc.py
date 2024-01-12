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

    time_interval: Time = Time.from_input(interval, TimeUnit.EPOCH)
    if time_interval.unit == TimeUnit.EPOCH:
        interval_event = Event.EPOCH_CHECKPOINT if checkpoint_events else Event.EPOCH_END
    elif time_interval.unit in {TimeUnit.BATCH, TimeUnit.TOKEN, TimeUnit.SAMPLE, TimeUnit.DURATION}:
        interval_event = Event.BATCH_CHECKPOINT if checkpoint_events else Event.BATCH_END
    else:
        raise NotImplementedError(
            f'Unknown interval: {time_interval.unit}. Must be TimeUnit.EPOCH, TimeUnit.BATCH, TimeUnit.TOKEN, or TimeUnit.SAMPLE.'
        )

    last_batch_seen = -1

    def check_interval(state: State, event: Event):
        # `TimeUnit.Duration` value is a float from `[0.0, 1.0)`
        if not time_interval.unit == TimeUnit.DURATION and int(time_interval) <= 0:
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

        if time_interval.unit in {TimeUnit.EPOCH, TimeUnit.BATCH, TimeUnit.TOKEN, TimeUnit.SAMPLE}:
            previous_count = state.previous_timestamp.get(time_interval.unit)
            count = state.timestamp.get(time_interval.unit)
        # If the eval_interval is a duration, we will track progress in terms of the unit of max_duration
        elif time_interval.unit == TimeUnit.DURATION:
            assert state.max_duration is not None
            previous_count = state.previous_timestamp.get(state.max_duration.unit)
            count = state.timestamp.get(state.max_duration.unit)
        else:
            raise NotImplementedError(
                f'Unknown interval: {time_interval.unit}. Must be TimeUnit.EPOCH, TimeUnit.BATCH, TimeUnit.TOKEN, or TimeUnit.SAMPLE.'
            )

        threshold_passed = math.floor(previous_count / time_interval.value) != math.floor(count / time_interval.value)

        if time_interval.unit != TimeUnit.DURATION and event == interval_event and threshold_passed:
            last_batch_seen = state.timestamp.batch
            return True
        elif time_interval.unit == TimeUnit.DURATION:
            assert state.max_duration is not None, 'max_duration should not be None'
            if state.dataloader_len is None:
                raise RuntimeError(
                    f'Interval of type `dur` or {TimeUnit.DURATION} requires the dataloader to be sized.')

            if event == interval_event:
                if state.max_duration.unit == TimeUnit.EPOCH and int(state.timestamp.batch) % math.ceil(
                        state.max_duration.value * float(time_interval) * state.dataloader_len) == 0:
                    last_batch_seen = state.timestamp.batch
                    return True
                elif state.max_duration.unit == TimeUnit.BATCH and int(state.timestamp.batch) % math.ceil(
                        state.max_duration.value * time_interval.value) == 0:
                    last_batch_seen = state.timestamp.batch
                    return True
                elif state.max_duration.unit == TimeUnit.SAMPLE:
                    samples_per_interval = math.ceil(state.max_duration.value * time_interval)
                    threshold_passed = math.floor(previous_count / samples_per_interval) != math.floor(
                        count / samples_per_interval)
                    if threshold_passed:
                        last_batch_seen = state.timestamp.batch
                        return True
                elif state.max_duration.unit == TimeUnit.TOKEN:
                    tokens_per_interval = math.ceil(state.max_duration.value * time_interval)
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


def partial_format(s, *args, **kwargs) -> str:
    """Format a string with a partial set of arguments.

    Since `str.format()` raises a `KeyError` if a format key is missing from the arguments, this
    function allows for a partial set of arguments to be provided. Any missing arguments will be
    left as-is in the string.
    """
    max_iters = 10_000  # Just in case we get stuck in a loop somehow.
    for _ in range(max_iters):
        try:
            return s.format(*args, **kwargs)
        except IndexError as e:  # Missing positional arg
            args += ('{}',)
        except KeyError as e:  # Missing keyword arg
            key = e.args[0]
            kwargs[key] = '{' + key + '}'

    raise RuntimeError(f'Failed to format string {s} after {max_iters} iterations.')
