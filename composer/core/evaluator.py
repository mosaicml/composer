# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""A wrapper for a dataloader to include metrics that apply to a specific dataset."""

from __future__ import annotations

import math
import warnings
from typing import Any, Callable, Dict, Iterable, List, Optional, Union

from torchmetrics import Metric, MetricCollection

from composer.core.data_spec import DataSpec, ensure_data_spec
from composer.core.event import Event
from composer.core.state import State
from composer.core.time import Time, TimeUnit

__all__ = ['Evaluator', 'evaluate_periodically', 'ensure_evaluator']


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

    if eval_interval.unit not in (TimeUnit.EPOCH, TimeUnit.BATCH, TimeUnit.DURATION):
        raise ValueError('The `eval_interval` must have units of EPOCH, BATCH, DURATION or be a function.')

    last_batch_seen = -1

    def should_eval(state: State, event: Event):
        # `TimeUnit.Duration` value is a float from `[0.0, 1.0)`
        if not eval_interval.unit == TimeUnit.DURATION and int(eval_interval) <= 0:
            return False
        nonlocal last_batch_seen  # required to use the last_batch_seen from the outer function scope

        # if requested, evaluate at the end of training, as long as the length of training is specified.
        if eval_at_fit_end and event == Event.FIT_END and state.timestamp.batch != last_batch_seen:
            return True

        if eval_interval.unit == TimeUnit.EPOCH and int(
                state.timestamp.epoch) % int(eval_interval) == 0 and event == Event.EPOCH_END:
            last_batch_seen = state.timestamp.batch
            return True

        if eval_interval.unit == TimeUnit.BATCH and int(
                state.timestamp.batch) % int(eval_interval) == 0 and event == Event.BATCH_END:
            last_batch_seen = state.timestamp.batch
            return True

        if eval_interval.unit == TimeUnit.DURATION:
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
                # If last sample in batch is not evenly divisible by eval_interval, perform evaluation in next batch
                if int(state.timestamp.batch) > 0:
                    samples_in_a_batch = int(state.timestamp.sample) // int(state.timestamp.batch)
                    if int(state.timestamp.sample) // math.ceil(state.max_duration.value * eval_interval) != int(
                            state.timestamp.sample - samples_in_a_batch) // math.ceil(
                                state.max_duration.value * eval_interval):
                        last_batch_seen = state.timestamp.batch
                        return True
            elif state.max_duration.unit == TimeUnit.TOKEN and event == Event.BATCH_END:
                raise ValueError(f'Evaluation interval of type `dur` is not supported yet for max_duration as `tok`')
        return False

    return should_eval


class Evaluator:
    """A wrapper for a dataloader to include metrics that apply to a specific dataset.

    For example, :class:`.CrossEntropyLoss` metric for NLP models.

    .. doctest::

       >>> eval_evaluator = Evaluator(
       ...     label="myEvaluator",
       ...     dataloader=eval_dataloader,
       ...     metric_names=['Accuracy']
       ... )
       >>> trainer = Trainer(
       ...     model=model,
       ...     train_dataloader=train_dataloader,
       ...     eval_dataloader=eval_evaluator,
       ...     optimizers=optimizer,
       ...     max_duration="1ep",
       ... )

    Args:
        label (str): Name of the Evaluator.
        dataloader (DataSpec | Iterable | Dict[str, Any]): Iterable that yields batches, a :class:`.DataSpec`
            for evaluation, or a Dict of :class:`.DataSpec` kwargs.
        metric_names: The list of metric names to compute.
            Each value in this list can be a regex string (e.g. "Accuracy", "f1" for "BinaryF1Score",
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
    """

    def __init__(
        self,
        *,
        label: str,
        dataloader: Union[DataSpec, Iterable, Dict[str, Any]],
        metric_names: Optional[List[str]] = None,
        metrics: Optional[Union[Metric, MetricCollection]] = None,
        subset_num_batches: Optional[int] = None,
        eval_interval: Optional[Union[int, str, Time, Callable[[State, Event], bool]]] = None,
    ):
        self.label = label
        self.dataloader = ensure_data_spec(dataloader)

        self.metric_names = []
        if metric_names is not None and metrics is not None:
            raise ValueError('only one of ``metrics`` or ``metric_names`` should be specified.')
        elif metric_names is not None:
            if not isinstance(metric_names, list):
                raise ValueError(f'``metric_names`` should be a list of strings, not a {type(metric_names)}')
            self.metric_names = metric_names
        elif metrics is not None:
            warnings.warn(DeprecationWarning('``metrics`` is deprecated and will be removed in 0.13.0.'))
            if isinstance(metrics, Metric):
                self.metric_names = [metrics.__class__.__name__]
            else:
                self.metric_names = [str(k) for k, _ in metrics.items()]

        self.subset_num_batches = subset_num_batches
        self._eval_interval = None
        self.eval_interval = eval_interval

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
