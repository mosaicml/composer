# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""A wrapper for a dataloader to include metrics that apply to a specific dataset."""

from __future__ import annotations

import math
import textwrap
import warnings
from typing import Any, Callable, Dict, Iterable, List, Optional, Union

from composer.core.data_spec import DataSpec, ensure_data_spec
from composer.core.event import Event
from composer.core.state import State
from composer.core.time import Time, TimeUnit
from composer.devices import Device, DeviceGPU

__all__ = ['Evaluator', 'evaluate_periodically', 'ensure_evaluator', 'validate_eval_automicrobatching']


def evaluate_periodically(eval_interval: Union[str, Time, int], eval_at_fit_end: bool = True):
    """Helper function to generate an evaluation interval callable.

    Args:
        eval_interval (str | Time | int): A :class:`.Time` instance or time string, or integer in epochs,
            representing how often to evaluate. Set to ``0`` to disable evaluation.
        eval_at_fit_end (bool): Whether to evaluate at the end of training, regardless of `eval_interval`.
            Default: True
    Returns:
        (State, Event) -> bool: A callable for the ``eval_interval`` argument of an
            :class:`.Evaluator`.
    """
    if isinstance(eval_interval, int):
        eval_interval = Time(eval_interval, TimeUnit.EPOCH)
    if isinstance(eval_interval, str):
        eval_interval = Time.from_timestring(eval_interval)

    last_batch_seen = -1

    def should_eval(state: State, event: Event):
        # `TimeUnit.Duration` value is a float from `[0.0, 1.0)`
        if not eval_interval.unit == TimeUnit.DURATION and int(eval_interval) <= 0:
            return False
        nonlocal last_batch_seen  # required to use the last_batch_seen from the outer function scope

        # if requested, evaluate at the end of training, as long as the length of training is specified.
        if eval_at_fit_end and event == Event.FIT_END and state.timestamp.batch != last_batch_seen:
            return True

        # Previous timestamp will only be None if training has not started, but we are returning False
        # in this case, just to be safe
        if state.previous_timestamp is None:
            return False

        if eval_interval.unit in {TimeUnit.EPOCH, TimeUnit.BATCH, TimeUnit.TOKEN, TimeUnit.SAMPLE}:
            previous_count = state.previous_timestamp.get(eval_interval.unit)
            count = state.timestamp.get(eval_interval.unit)
        # If the eval_interval is a duration, we will track progress in terms of the unit of max_duration
        elif eval_interval.unit == TimeUnit.DURATION:
            assert state.max_duration is not None
            previous_count = state.previous_timestamp.get(state.max_duration.unit)
            count = state.timestamp.get(state.max_duration.unit)
        else:
            raise ValueError(f'Invalid eval_interval unit: {eval_interval.unit}')

        threshold_passed = math.floor(previous_count / eval_interval.value) != math.floor(count / eval_interval.value)

        if eval_interval.unit == TimeUnit.EPOCH and event == Event.EPOCH_END and threshold_passed:
            last_batch_seen = state.timestamp.batch
            return True
        elif eval_interval.unit in {TimeUnit.BATCH, TimeUnit.TOKEN, TimeUnit.SAMPLE
                                   } and event == Event.BATCH_END and threshold_passed:
            last_batch_seen = state.timestamp.batch
            return True
        elif eval_interval.unit == TimeUnit.DURATION:
            assert state.max_duration is not None, 'max_duration should not be None'
            if state.dataloader_len is None:
                raise RuntimeError(
                    f'Evaluation interval of type `dur` or {TimeUnit.DURATION} requires the dataloader to be sized.')
            if state.max_duration.unit == TimeUnit.EPOCH and int(
                    state.timestamp.batch) % math.ceil(state.max_duration.value * float(eval_interval) *
                                                       state.dataloader_len) == 0 and event == Event.BATCH_END:
                last_batch_seen = state.timestamp.batch
                return True
            elif state.max_duration.unit == TimeUnit.BATCH and int(state.timestamp.batch) % math.ceil(
                    state.max_duration.value * eval_interval.value) == 0 and event == Event.BATCH_END:
                last_batch_seen = state.timestamp.batch
                return True
            elif state.max_duration.unit == TimeUnit.SAMPLE and event == Event.BATCH_END:
                samples_per_interval = math.ceil(state.max_duration.value * eval_interval)
                threshold_passed = math.floor(previous_count / samples_per_interval) != math.floor(
                    count / samples_per_interval)
                if threshold_passed:
                    last_batch_seen = state.timestamp.batch
                    return True
            elif state.max_duration.unit == TimeUnit.TOKEN and event == Event.BATCH_END:
                tokens_per_interval = math.ceil(state.max_duration.value * eval_interval)
                threshold_passed = math.floor(previous_count / tokens_per_interval) != math.floor(
                    count / tokens_per_interval)
                if threshold_passed:
                    last_batch_seen = state.timestamp.batch
                    return True
        return False

    return should_eval


class Evaluator:
    """A wrapper for a dataloader to include metrics that apply to a specific dataset.

    For example, :class:`.CrossEntropyLoss` metric for NLP models.

    .. doctest::

       >>> eval_evaluator = Evaluator(
       ...     label='myEvaluator',
       ...     dataloader=eval_dataloader,
       ...     metric_names=['MulticlassAccuracy']
       ... )
       >>> trainer = Trainer(
       ...     model=model,
       ...     train_dataloader=train_dataloader,
       ...     eval_dataloader=eval_evaluator,
       ...     optimizers=optimizer,
       ...     max_duration='1ep',
       ... )

    Args:
        label (str): Name of the Evaluator.
        dataloader (DataSpec | Iterable | Dict[str, Any]): Iterable that yields batches, a :class:`.DataSpec`
            for evaluation, or a Dict of :class:`.DataSpec` kwargs.
        metric_names: The list of metric names to compute.
            Each value in this list can be a regex string (e.g. "MulticlassAccuracy", "f1" for "BinaryF1Score",
            "Top-." for "Top-1", "Top-2", etc). Each regex string will be matched against the keys of the dictionary returned
            by ``model.get_metrics()``. All matching metrics will be evaluated.

            By default, if left blank, then all metrics returned by ``model.get_metrics()`` will be used.
        subset_num_batches (int, optional): The maximum number of batches to use for each evaluation. Defaults to ``None``,
            which means that the ``eval_subset_num_batches`` parameter from the :class:`.Trainer` will be used.
            Set to ``-1`` to evaluate the entire ``dataloader``.
        eval_interval (Time | int | str | (State, Event) -> bool, optional): An integer,
            which will be interpreted to be epochs, a str (e.g. ``1ep``, or ``10ba``), a :class:`.Time` object, or a callable.
            Defaults to ``None``, which means that the ``eval_interval`` parameter from the :class:`.Trainer` will be used.

            If an integer (in epochs), :class:`.Time` string, or :class:`.Time` instance, the evaluator will be run
            with this frequency. :class:`.Time` strings or :class:`.Time` instances must have units of
            :attr:`.TimeUnit.BATCH` or :attr:`.TimeUnit.EPOCH`.

            Set to ``0`` to disable evaluation.

            If a callable, it should take two arguments (:class:`.State`, :class:`.Event`) and return a bool
            representing whether the evaluator should be invoked. The event will be either :attr:`.Event.BATCH_END`
            or :attr:`.Event.EPOCH_END`.

            When specifying ``eval_interval``, the evaluator(s) are also run at the ``Event.FIT_END`` if it doesn't
            evenly divide the training duration.
        device_eval_microbatch_size (int, optional): The number of samples to use for each microbatch when evaluating.
            If set to ``auto``, dynamically decreases device_eval_microbatch_size if microbatch is too large for GPU.
            If None, sets `device_eval_microbatch_size` to per rank batch size. (default: ``None``)
    """

    def __init__(
        self,
        *,
        label: str,
        dataloader: Union[DataSpec, Iterable, Dict[str, Any]],
        metric_names: Optional[List[str]] = None,
        subset_num_batches: Optional[int] = None,
        eval_interval: Optional[Union[int, str, Time, Callable[[State, Event], bool]]] = None,
        device_eval_microbatch_size: Optional[Union[int, str]] = None,
    ):
        self.label = label
        self.dataloader = ensure_data_spec(dataloader)

        self.metric_names = []
        if metric_names is not None:
            if not isinstance(metric_names, list):
                raise ValueError(f'``metric_names`` should be a list of strings, not a {type(metric_names)}')
            self.metric_names = metric_names

        self.subset_num_batches = subset_num_batches
        self._eval_interval = None
        self.eval_interval = eval_interval
        self.auto_microbatching = _is_auto_microbatching(device_eval_microbatch_size)
        self.device_eval_microbatch_size = _get_initial_device_eval_microbatch_size(
            device_eval_microbatch_size,
            self.auto_microbatching,
            self.dataloader.dataloader,
        )

    @property
    def eval_interval(self):
        return self._eval_interval

    @eval_interval.setter
    def eval_interval(self, eval_interval: Optional[Union[int, str, Time, Callable[[State, Event], bool]]]):
        if eval_interval is None:
            self._eval_interval = None
        elif not callable(eval_interval):
            self._eval_interval = evaluate_periodically(eval_interval)
        else:
            self._eval_interval = eval_interval


def ensure_evaluator(evaluator: Union[Evaluator, DataSpec, Iterable, Dict[str, Any]], default_metric_names: List[str]):
    """Ensure that ``evaluator`` is an :class:`.Evaluator`.

    Args:
        evaluator (Evaluator | DataSpec | Iterable | Dict[str, Any]): A dataloader,
            :class:`.DataSpec` instance, dictionary of :class:`.DataSpec` kwargs, or existing evaluator.
        default_metric_names (List[str]): The names of the metrics for the ``evaluator``,
            if a dataloader was specified.

    Returns:
        Evaluator: An evaluator.
    """
    if isinstance(evaluator, Evaluator):
        return evaluator
    else:
        return Evaluator(
            label='eval',
            dataloader=evaluator,
            metric_names=default_metric_names,
        )


def validate_eval_automicrobatching(auto_microbatching: bool, device: Device):
    """Ensure automicrobatching is only on GPU.

    Unlike `device_train_microbatch_size`, this validation must be done separately from the
    `_is_auto_microbatching` check because `device` is not available during `Evaluator`
    initialization.
    """
    if auto_microbatching and not isinstance(device, DeviceGPU):
        raise ValueError(
            'Can only use adaptive device_eval_microbatch_size on GPU. Please set device_eval_microbatch_size >= 1.')


def _is_auto_microbatching(device_eval_microbatch_size: Optional[Union[int, str]]):
    if device_eval_microbatch_size == 'auto':
        warnings.warn(("Setting `device_eval_microbatch_size='auto'` is an experimental feature which may cause "
                       'uncaught Cuda Out of Memory errors. In this case, please manually '
                       'set device_eval_microbatch_size explicitly to an integer instead.'))
        return True
    else:
        return False


def _get_initial_device_eval_microbatch_size(device_eval_microbatch_size: Optional[Union[int, str]],
                                             auto_microbatching: bool, dataloader: Iterable) -> int:
    """Sets initial value of device_eval_microbatch_size.

    If auto_microbatching, sets initial `device_eval_microbatch_size` to per rank batch size.
    """
    if auto_microbatching or device_eval_microbatch_size is None:
        try:
            batch_size = getattr(dataloader, 'batch_size')
        except AttributeError as e:
            if auto_microbatching:
                raise AttributeError(
                    "`device_eval_microbatch_size='auto'` requires the `dataloader` to have a `batch_size` attribute."
                ) from e
            else:
                raise AttributeError(
                    textwrap.dedent(
                        '`device_eval_microbatch_size` is not set and `dataloader` does not have a `batch_size` attribute. '
                        'Please either set `device_eval_microbatch_size` or `dataloader.batch_size`.')) from e
        return batch_size
    elif isinstance(device_eval_microbatch_size, int):
        return device_eval_microbatch_size
    else:
        raise ValueError("device_eval_microbatch_size must be an int or ``'auto'``")
