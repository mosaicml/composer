# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Hyperparameters for the :class:`~.evaluator.Evaluator`."""

from __future__ import annotations

import logging
import re
import textwrap
from dataclasses import dataclass
from typing import List, Optional

import yahp as hp

from composer.core.evaluator import Evaluator
from composer.datasets.dataset_hparams import DataLoaderHparams, DatasetHparams
from composer.datasets.dataset_hparams_registry import dataset_registry
from composer.models.base import ComposerModel

log = logging.getLogger(__name__)

__all__ = ['EvaluatorHparams']


@dataclass
class EvaluatorHparams(hp.Hparams):
    """Params for the :class:`~.evaluator.Evaluator`.

    Also see the documentation for the :class:`~.evaluator.Evaluator`.

    Args:
        label (str): Name of the Evaluator. Used for logging/reporting metrics.
        eval_interval (str, optional): See :class:`.Evaluator`.
        subset_num_batches (int, optional): See :class:`.Evaluator`.
        eval_dataset (DatasetHparams): Evaluation dataset.
        metric_names (List[str], optional): List of names of the metrics for the
            evaluator. Can be a :class:`torchmetrics.Metric` name or the class name of a
            metric returned by :meth:`~.ComposerModel.metrics` If
            ``None``, uses all metrics in the model. Default: ``None``.
    """
    hparams_registry = {
        'eval_dataset': dataset_registry,
    }

    label: str = hp.auto(Evaluator, 'label')
    eval_dataset: DatasetHparams = hp.required(doc='Evaluator dataset for the Evaluator')
    eval_interval: Optional[str] = hp.auto(Evaluator, 'eval_interval')
    subset_num_batches: Optional[int] = hp.auto(Evaluator, 'subset_num_batches')
    metric_names: Optional[List[str]] = hp.optional(
        doc=textwrap.dedent("""Name of the metrics for the evaluator. Can be a torchmetrics metric name or the
        class name of a metric returned by model.get_metrics(). If None (the default), uses all metrics in the model"""
                           ),
        default=None)

    def initialize_object(self, model: ComposerModel, batch_size: int, dataloader_hparams: DataLoaderHparams):
        """Initialize an :class:`~.evaluator.Evaluator`

        If the Evaluator ``metric_names`` is empty or None is provided, the function
        returns a copy of all the model's default evaluation metrics.

        Args:
            model (ComposerModel): The model, which is used to retrieve metric names.
            batch_size (int): The device batch size to use for the evaluation dataset.
            dataloader_hparams (DataLoaderHparams): The hparams to use to construct a dataloader for the evaluation dataset.

        Returns:
            Evaluator: The evaluator.
        """
        dataloader = self.eval_dataset.initialize_object(batch_size=batch_size, dataloader_hparams=dataloader_hparams)

        # Get and copy all the model's associated evaluation metrics
        model_metrics = model.get_metrics(is_train=False)

        # Use all the metrics from the model if no metric_names are specified
        if self.metric_names is None:
            evaluator_metric_names = [str(k) for k in model_metrics.keys()]  # convert hashable to str
        else:
            evaluator_metric_names = []
            for metric_name in self.metric_names:
                if not any(re.match(f'.*{metric_name}.*', k, re.IGNORECASE) for k in model_metrics.keys()):
                    raise RuntimeError(
                        textwrap.dedent(f"""No metric found with the name {metric_name}. Check if this"
                                       "metric is compatible/listed in your model metrics."""))
                evaluator_metric_names.append(str(metric_name))
            if len(evaluator_metric_names) == 0:
                raise RuntimeError(
                    textwrap.dedent(f"""No metrics compatible with your model were added to this evaluator.
                    Check that the metrics you specified are compatible/listed in your model."""))

        return Evaluator(
            label=self.label,
            dataloader=dataloader,
            metric_names=evaluator_metric_names,
            eval_interval=self.eval_interval,
            subset_num_batches=self.subset_num_batches,
        )
