# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""A wrapper for a dataloader to include metrics that apply to a specific dataset."""

from __future__ import annotations

import textwrap
import warnings
from typing import Any, Callable, Iterable, Optional, Union

from composer.core.data_spec import DataSpec, ensure_data_spec
from composer.core.event import Event
from composer.core.state import State
from composer.core.time import Time
from composer.utils import create_interval_scheduler

__all__ = ['Evaluator', 'ensure_evaluator']


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
        dataloader (DataSpec | Iterable | dict[str, Any]): Iterable that yields batches, a :class:`.DataSpec`
            for evaluation, or a dict of :class:`.DataSpec` kwargs.
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
        device_eval_microbatch_size (str | int | float, optional): The number of samples to use for each microbatch when evaluating.
            If set to ``auto``, dynamically decreases device_eval_microbatch_size if microbatch is too large for GPU.
            If None, sets `device_eval_microbatch_size` to per rank batch size. (default: ``None``)
    """

    def __init__(
        self,
        *,
        label: str,
        dataloader: Union[DataSpec, Iterable, dict[str, Any]],
        metric_names: Optional[list[str]] = None,
        subset_num_batches: Optional[int] = None,
        eval_interval: Optional[Union[int, str, Time, Callable[[State, Event], bool]]] = None,
        device_eval_microbatch_size: Optional[Union[int, str, float]] = None,
    ):
        self.label = label
        self.dataloader = ensure_data_spec(dataloader)

        if metric_names is not None:
            if not isinstance(metric_names, list):
                raise ValueError(f'``metric_names`` should be a list of strings, not a {type(metric_names)}')
        self.metric_names = metric_names

        self.subset_num_batches = subset_num_batches
        self._eval_interval = None
        self.eval_interval = eval_interval
        self.auto_microbatching = _is_auto_microbatching(device_eval_microbatch_size)
        if self.auto_microbatching and hasattr(self.dataloader, 'seq_parallel_world_size'):
            raise ValueError('`device_eval_microbatch_size="auto"` is not compatible with sequence parallelism.')
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
            self._eval_interval = create_interval_scheduler(
                eval_interval,
                checkpoint_events=False,
                final_events={Event.FIT_END},
            )
        else:
            self._eval_interval = eval_interval


def ensure_evaluator(evaluator: Union[Evaluator, DataSpec, Iterable, dict[str, Any]], default_metric_names: list[str]):
    """Ensure that ``evaluator`` is an :class:`.Evaluator`.

    Args:
        evaluator (Evaluator | DataSpec | Iterable | dict[str, Any]): A dataloader,
            :class:`.DataSpec` instance, dictionary of :class:`.DataSpec` kwargs, or existing evaluator.
        default_metric_names (list[str]): The names of the metrics for the ``evaluator``,
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


def _is_auto_microbatching(device_eval_microbatch_size: Optional[Union[int, str, float]]):
    if device_eval_microbatch_size == 'auto':
        warnings.warn((
            "Setting `device_eval_microbatch_size='auto'` is an experimental feature which may cause "
            'uncaught Cuda Out of Memory errors. In this case, please manually '
            'set device_eval_microbatch_size explicitly to an integer instead.'
        ))
        return True
    else:
        return False


def _get_initial_device_eval_microbatch_size(
    device_eval_microbatch_size: Optional[Union[int, str, float]],
    auto_microbatching: bool,
    dataloader: Iterable,
) -> Union[int, float]:
    """Sets initial value of device_eval_microbatch_size.

    If auto_microbatching, sets initial `device_eval_microbatch_size` to per rank batch size.
    """
    if auto_microbatching or device_eval_microbatch_size is None:
        try:
            batch_size = getattr(dataloader, 'batch_size')
        except AttributeError as e:
            if auto_microbatching:
                raise AttributeError(
                    "`device_eval_microbatch_size='auto'` requires the `dataloader` to have a `batch_size` attribute.",
                ) from e
            else:
                raise AttributeError(
                    textwrap.dedent(
                        '`device_eval_microbatch_size` is not set and `dataloader` does not have a `batch_size` attribute. '
                        'Please either set `device_eval_microbatch_size` or `dataloader.batch_size`.',
                    ),
                ) from e
        return batch_size
    elif isinstance(device_eval_microbatch_size, (int, float)):
        return device_eval_microbatch_size
    else:
        raise ValueError("device_eval_microbatch_size must be an int or ``'auto'``")
