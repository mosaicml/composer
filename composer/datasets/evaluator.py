# Copyright 2021 MosaicML. All Rights Reserved.

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List

import yahp as hp
from torchmetrics.collections import MetricCollection

from composer.core.types import Evaluator
from composer.datasets.dataset_registry import get_dataset_registry
from composer.datasets.hparams import DatasetHparams
from composer.datasets import DataloaderHparams
from typing import Optional

log = logging.getLogger(__name__)



@dataclass
class EvaluatorHparams(hp.Hparams):
    hparams_registry = {  # type: ignore
        "eval_dataset": get_dataset_registry(),
    }

    label: str = hp.required(doc="Name of the Evaluator object. Used for logging/reporting metrics")
    eval_dataset: DatasetHparams = hp.required(doc="Evaluator dataset for the Evaluator")
    validate_every_n_epochs: int = hp.optional(
        doc="Validate every N epochs. Set to -1 to never validate on a epochwise frequency. Defaults to 1", default=1)
    validate_every_n_batches: int = hp.optional(
        doc="Validate every N batches. Set to -1 to never validate on a batchwise frequency. Defaults to -1.",
        default=-1)
    metric_names: Optional[List[str]] = hp.optional(doc="Name of the metrics for the evaluator.Use the torchmetrics"
                    "metric name for torchmetrics and use the classname for custom metrics.", default_factory=list)

    def initialize_object(self, batch_size: int, dataloader_hparams: DataloaderHparams):
        dataloader = self.eval_dataset.initialize_object(batch_size=batch_size, dataloader_hparams=dataloader_hparams)

        # Populate the metrics later in the trainer initialization
        return Evaluator(label=self.label, dataloader=dataloader, metrics=MetricCollection([]), metric_names=self.metric_names, validate_every_n_batches=self.validate_every_n_batches, validate_every_n_epochs=self.validate_every_n_epochs)
