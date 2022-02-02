# Copyright 2021 MosaicML. All Rights Reserved.

from __future__ import annotations

import copy
import logging
import textwrap
from dataclasses import dataclass
from typing import List, Optional

import yahp as hp
from torchmetrics import Metric, MetricCollection

from composer.core.types import Evaluator
from composer.datasets import DataloaderHparams
from composer.datasets.dataset_registry import get_dataset_registry
from composer.datasets.hparams import DatasetHparams
from composer.models.base import ComposerModel

log = logging.getLogger(__name__)


@dataclass
class EvaluatorHparams(hp.Hparams):
    """Params for the :class:`Evaluator`.

    See the documentation for the :class:`Evaluator`.
    """
    hparams_registry = {  # type: ignore
        "eval_dataset": get_dataset_registry(),
    }

    label: str = hp.required(doc="Name of the Evaluator object. Used for logging/reporting metrics")
    eval_dataset: DatasetHparams = hp.required(doc="Evaluator dataset for the Evaluator")
    metric_names: Optional[List[str]] = hp.optional(
        doc=textwrap.dedent("""Name of the metrics for the evaluator. Can be a torchmetrics metric name or the
        class name of a metric returned by model.metrics(). If None (the default), uses all metrics in the model"""),
        default=None)

    def initialize_object(self, model: ComposerModel, batch_size: int, dataloader_hparams: DataloaderHparams):
        """Initialize an :class:`Evaluator`

        If the Evaluatormetric_names is empty or None is provided, the function returns
        a copy of all the model's default evaluation metrics.

        Args:
            model (ComposerModel): The model, which is used to retrieve metric names
            batch_size (int): The device batch size to use for the evaluation dataset
            dataloader_hparams (DataloaderHparams): The hparams to use to construct a dataloader for the evaluation dataset

        Returns:
            Evaluator: The evaluator
        """
        dataloader = self.eval_dataset.initialize_object(batch_size=batch_size, dataloader_hparams=dataloader_hparams)

        # Get and copy all the model's associated evaluation metrics
        model_metrics = model.metrics(train=False)
        if isinstance(model_metrics, Metric):
            # Forcing metrics to be a MetricCollection simplifies logging results
            model_metrics = MetricCollection([model_metrics])

        # Use all the metrics from the model if no metric_names are specified
        if self.metric_names is None:
            evaluator_metrics = copy.deepcopy(model_metrics)
        else:
            evaluator_metrics = MetricCollection([])
            for metric_name in self.metric_names:
                try:
                    metric = model_metrics[metric_name]
                except KeyError as e:
                    raise RuntimeError(
                        textwrap.dedent(f"""No metric found with the name {metric_name}. Check if this"
                                       "metric is compatible/listed in your model metrics.""")) from e
                assert isinstance(metric, Metric), "all values of a MetricCollection.__getitem__ should be a metric"
                evaluator_metrics.add_metrics(copy.deepcopy(metric))
            if len(evaluator_metrics) == 0:
                raise RuntimeError(
                    textwrap.dedent(f"""No metrics compatible with your model were added to this evaluator.
                    Check that the metrics you specified are compatible/listed in your model."""))

        return Evaluator(
            label=self.label,
            dataloader=dataloader,
            metrics=evaluator_metrics,
        )
