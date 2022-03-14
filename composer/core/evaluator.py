# Copyright 2021 MosaicML. All Rights Reserved.

"""A wrapper for a dataloader to include metrics that apply to a specific dataset."""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Union

from torchmetrics import Metric, MetricCollection

from composer.core.data_spec import DataSpec as DataSpec

if TYPE_CHECKING:
    from composer.core.types import DataLoader, Metrics

__all__ = ["Evaluator"]


class Evaluator:
    """A wrapper for a dataloader to include metrics that apply to a specific dataset.

    For example, :class:`~.nlp_metrics.CrossEntropyLoss` metric for NLP models.

       >>> from torchmetrics.classification.accuracy import Accuracy
       >>> eval_evaluator = Evaluator(label="myEvaluator", dataloader=eval_dataloader, metrics=Accuracy())
       >>> trainer = Trainer(
       ...     model=model,
       ...     train_dataloader=train_dataloader,
       ...     eval_dataloader=eval_evaluator,
       ...     optimizers=optimizer,
       ...     max_duration="1ep",
       ... )


    Args:
        label (str): Name of the Evaluator
        dataloader (Union[DataSpec, DataLoader]): DataLoader/DataSpec for evaluation data
        metrics (Metrics): :class:`torchmetrics.Metric` to log. ``metrics`` will be deep-copied to ensure
            that each evaluator updates only its ``metrics``.
    """

    def __init__(self, *, label: str, dataloader: Union[DataSpec, DataLoader], metrics: Metrics):
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
