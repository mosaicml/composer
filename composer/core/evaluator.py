# Copyright 2021 MosaicML. All Rights Reserved.

"""A wrapper for a dataloader to include metrics that apply to a specific dataset."""

from __future__ import annotations

import copy
from typing import Callable, Iterable, Optional, Union

from torchmetrics import Metric, MetricCollection

from composer.core.data_spec import DataSpec
from composer.core.event import Event
from composer.core.state import State
from composer.core.time import Time, TimeUnit

__all__ = ["Evaluator", "evaluate_periodically"]


def evaluate_periodically(eval_interval: Union[str, Time, int]):
    """Helper function to generate an evaluation interval callable.

    Args:
        eval_interval (str | Time | int): A :class:`.Time` instance or time string, or integer in epochs,
            representing how often to evaluate. Set to ``0`` to disable evaluation.
    Returns:
        (State, Event) -> bool: A callable for the ``eval_interval`` argument of an :class:`.Evaluator`.
    """
    if isinstance(eval_interval, int):
        eval_interval = Time(eval_interval, TimeUnit.EPOCH)
    if isinstance(eval_interval, str):
        eval_interval = Time.from_timestring(eval_interval)

    if eval_interval.unit not in (TimeUnit.EPOCH, TimeUnit.BATCH):
        raise ValueError("The `eval_interval` must have units of EPOCH or BATCH, or be a function.")

    def should_eval(state: State, event: Event):
        if int(eval_interval) <= 0:
            return False

        if eval_interval.unit == TimeUnit.EPOCH:
            return int(state.timer.epoch) % int(eval_interval) == 0 and event == Event.EPOCH_END
        if eval_interval.unit == TimeUnit.BATCH:
            return int(state.timer.batch) % int(eval_interval) == 0 and event == Event.BATCH_END

        return False

    return should_eval


class Evaluator:
    """A wrapper for a dataloader to include metrics that apply to a specific dataset.

    For example, :class:`~.nlp_metrics.CrossEntropyLoss` metric for NLP models.

    .. doctest::

       >>> from torchmetrics.classification.accuracy import Accuracy
       >>> eval_evaluator = Evaluator(label="myEvaluator", dataloader=eval_dataloader, metrics=Accuracy())
       >>> trainer = Trainer(
       ...     model=model,
       ...     train_dataloader=train_dataloader,
       ...     eval_dataloader=eval_evaluator,
       ...     optimizers=optimizer,
       ...     max_duration="1ep",
       ... )

    .. testcleanup::

        trainer.engine.close()


    Args:
        label (str): Name of the Evaluator
        dataloader (Union[DataSpec, Iterable]): Iterable that yields batches or a :class:`.DataSpec` for evaluation
            data.
        metrics (Metric | MetricCollection): :class:`torchmetrics.Metric` to log. ``metrics`` will be deep-copied to
            ensure that each evaluator updates only its ``metrics``.
        subset_num_batches (int, optional): The maximum number of batches to use for each evaluation. Defaults to
            ``None``, which means that the ``eval_subset_num_batches`` parameter from the
            :class:`~composer.trainer.trainer.Trainer` will be used.

            Set to ``-1`` to evaluate the entire ``dataloader``
        eval_interval (int | str | Time | (State, Event) -> bool, optional): An integer, which will be
            interpreted to be epochs, a str (e.g. ``1ep``, or ``10ba``), a :class:`.Time` object, or a callable.
            Defaults to ``None``, which means that the ``eval_interval`` parameter from the
            :class:`~composer.trainer.trainer.Trainer` will be used.

            If an integer (in epochs), :class:`.Time` string, or :class:`.Time` instance, the evaluator will be run
            with this frequency. :class:`.Time` strings or :class:`.Time` instances must have units of
            :attr:`.TimeUnit.BATCH` or :attr:`.TimeUnit.EPOCH`.

            Set to ``0`` to disable evaluation.

            If a callable, it should take two arguments (:class:`.State`, :class:`.Event`) and return a bool 
            representing whether the evaluator should be invoked. The event will be either :attr:`.Event.BATCH_END`
            or :attr:`.Event.EPOCH_END`.
    """

    def __init__(
        self,
        *,
        label: str,
        dataloader: Union[DataSpec, Iterable],
        metrics: Union[Metric, MetricCollection],
        subset_num_batches: Optional[int] = None,
        eval_interval: Optional[Union[int, str, Time, Callable[[State, Event], bool]]] = None,
    ):
        self.label = label
        if isinstance(dataloader, DataSpec):
            self.dataloader = dataloader
        else:
            self.dataloader = DataSpec(dataloader)

        # Forcing metrics to be a MetricCollection simplifies logging results
        metrics = copy.deepcopy(metrics)
        if isinstance(metrics, Metric):
            self.metrics = MetricCollection([metrics])
        else:
            self.metrics = metrics

        self.subset_num_batches = subset_num_batches

        if eval_interval is None:
            self.should_eval = None
        elif not callable(eval_interval):
            self.should_eval = evaluate_periodically(eval_interval)
        else:
            self.should_eval = eval_interval
