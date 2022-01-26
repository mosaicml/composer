# Copyright 2021 MosaicML. All Rights Reserved.

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional

import yahp as hp

from composer.core.types import Evaluator
from composer.datasets import DataloaderHparams
from composer.datasets.dataset_registry import get_dataset_registry
from composer.datasets.hparams import DatasetHparams

log = logging.getLogger(__name__)


@dataclass
class EvaluatorHparams(hp.Hparams):
    hparams_registry = {  # type: ignore
        "eval_dataset": get_dataset_registry(),
    }

    label: str = hp.required(doc="Name of the Evaluator object. Used for logging/reporting metrics")
    eval_dataset: DatasetHparams = hp.required(doc="Evaluator dataset for the Evaluator")
    metric_names: Optional[List[str]] = hp.optional(
        doc="Name of the metrics for the evaluator.Use the torchmetrics"
        "metric name for torchmetrics and use the classname for custom metrics.",
        default_factory=list)
    eval_subset_num_batches: Optional[int] = hp.optional("If specified, evaluate on this many batches.", default=None)

    def initialize_object(self, batch_size: int, dataloader_hparams: DataloaderHparams):
        dataloader = self.eval_dataset.initialize_object(batch_size=batch_size, dataloader_hparams=dataloader_hparams)

        # Populate the metrics later in the trainer initialization
        return Evaluator(
            label=self.label,
            dataloader=dataloader,
            metrics=self.metric_names,
        )
