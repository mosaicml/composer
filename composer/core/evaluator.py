# Copyright 2021 MosaicML. All Rights Reserved.

"""A wrapper for a dataloader to include metrics that apply to a specific dataset."""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Union, Callable

from torchmetrics import Metric, MetricCollection

from composer.core.data_spec import DataSpec
from composer.core.time import Time, TimeUnit
from composer.core.state import State
from composer.core.event import Event

if TYPE_CHECKING:
    from composer.core.types import DataLoader

__all__ = ["Evaluator"]


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
        dataloader (Union[DataSpec, DataLoader]): DataLoader/DataSpec for evaluation data
        metrics (Metric | MetricCollection): :class:`torchmetrics.Metric` to log. ``metrics`` will be deep-copied to ensure
            that each evaluator updates only its ``metrics``.
        subset_num_batches (int, optional): The maximum number of batches to use for each evaluation. Set to ``-1`` to
            evaluate the entire ``dataloader``. Defaults to ``-1``.
        interval (int | str | Time | (State, Event) -> bool, optional): An integer (in epochs),
            :class:`.Time` string or instance, or a callable that takes the (State, Event) and returns whether to
            evaluate the evaluator. Defaults to ``1`` (evaluate every epoch).

            If an integer (in epochs), :class:`.Time` string, or :class:`.Time` instance, the evaluator will be run
            with this frequency. :class:`.Time` strings or :class:`.Time` instances must have units of
            :attr:`.TimeUnit.BATCH` or :attr:`.TimeUnit.EPOCH`.

            If a callable, it will be called with the training :class:`.State` and the evaluation event, which will be
            either :attr:`.Event.BATCH_END` or :attr:`.Event.EPOCH_END`. The callable should return a bool representing
            whether the evaluator should be invoked.
    """

    def __init__(
        self,
        *,
        label: str,
        dataloader: Union[DataSpec, DataLoader],
        metrics: Union[Metric, MetricCollection],
        subset_num_batches: int = -1,
        interval: Union[int, str, Time, Callable[[State, Event], bool]] = 1,
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

        if isinstance(interval, int):
            interval = Time(interval, TimeUnit.EPOCH)
        if isinstance(interval, str):
            interval = Time.from_timestring(interval)
        if isinstance(interval, Time):
            if interval.unit not in (TimeUnit.EPOCH, TimeUnit.BATCH):
                raise ValueError("The `interval` must have units of EPOCH or BATCH, or be a function.")

            def interval_callable(state: State, event: Event, interval: Time[int] = interval):
                if interval.unit == TimeUnit.EPOCH:
                    return int(state.timer.epoch) % int(interval) == 0 and event == Event.EPOCH_END
                if interval.unit == TimeUnit.BATCH:
                    return int(state.timer.batch) % int(interval) == 0 and event == Event.BATCH_END

                return False

            interval = interval_callable

        self.interval = interval
