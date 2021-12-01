# Copyright 2021 MosaicML. All Rights Reserved.

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List

import yahp as hp
from torchmetrics.classification.accuracy import Accuracy
from torchmetrics.classification.f_beta import F1
from torchmetrics.classification.matthews_corrcoef import MatthewsCorrcoef
from torchmetrics.collections import MetricCollection
from torchmetrics.metric import Metric
from torchmetrics.regression.pearson import PearsonCorrcoef
from torchmetrics.regression.spearman import SpearmanCorrcoef

from composer.core.types import Metrics
from composer.datasets.dataset_registry import get_dataset_registry
from composer.datasets.hparams import DataloaderSpec, DatasetHparams
from composer.models.loss import CrossEntropyLoss

log = logging.getLogger(__name__)


@dataclass
class EvaluatorHparams(hp.Hparams):
    hparams_registry = {  # type: ignore
        "eval_dataset": get_dataset_registry(),
    }

    metric_registry = {
        "Accuracy": Accuracy,
        "MatthewsCorrcoef": MatthewsCorrcoef,
        "F1": F1,
        "PearsonCorrcoef": PearsonCorrcoef,
        "SpearmanCorrcoef": SpearmanCorrcoef,
        "CrossEntropyLoss": CrossEntropyLoss,
    }

    label: str = hp.required(doc="label")
    eval_dataset: DatasetHparams = hp.required(doc="Training dataset hparams")
    metrics: List[str] = hp.required(doc="metrics", template_default=list)

    def initialize_object(self):
        dataset = self.eval_dataset.initialize_object()

        def resolve_metric_list(metric_names: List[str]) -> List[Metric]:
            resolved_metrics = []
            for metric_name in metric_names:
                if metric_name in self.metric_registry:
                    metric_instance = self.metric_registry[metric_name]()
                    resolved_metrics.append(metric_instance)
                else:
                    log.warning(f"The metric {metric_name} is not a registered metric. Add it to the metric registry.")
            return resolved_metrics

        metric_list = resolve_metric_list(self.metrics)
        return EvaluatorSpec(label=self.label, dataloader_spec=dataset, metrics=MetricCollection(metric_list))


@dataclass
class EvaluatorSpec:
    """Wrapper class to contain DataLoaderSpecs and relevant metrics to use during evaluation.

    Args:
        label (str): This is the evaluator/dataset label that gets used for reporting metrics
        dataloader_spec (DataloaderSpec): DataloaderSpec specifying the dataset to use for
            the evaluator.
        metric_list: (List[Metric]) 
    """
    label: str
    dataloader_spec: DataloaderSpec
    metrics: Metrics
