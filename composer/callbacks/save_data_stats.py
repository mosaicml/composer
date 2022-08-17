# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Threshold stopping callback."""

import copy
import os
import pickle
import tempfile
from typing import Any, Collection, Dict, Iterable, List, Union

from torchmetrics import Metric, MetricCollection

from composer.core import State
from composer.core.callback import Callback
from composer.core.data_spec import DataSpec, ensure_data_spec
from composer.loggers import Logger, LogLevel
from composer.utils import dist

# import torch

__all__ = ['DataStatSaver']


class DataStatSaver(Callback):
    """Save stats about the training data at the end of training

    Args:
        filename (str): Name of file to save.
        save_destination (str): Path
        metrics (List[str]): Metrics to save
    """

    def __init__(self, filename: str, save_destination: str, metrics: Dict[str, str]):
        self.training_finished = False
        if '.pkl' not in filename:
            filename += '.pkl'
        self.filename = filename
        self.save_destination = save_destination
        self.metrics = metrics

    def collate_sample_metrics(self, metrics: Dict) -> Dict[str, Dict[str, Any]]:
        """Aggregates metrics into a single dictionary in which each key corresponds to a data
        sample uid and each value is a dictionary of metrics and values.

        Args:
            metrics (Collection): Metrics to be aggregated.

        Returns:
            sample_metrics (Dict[str, Dict[str, Any]]): Dictionary of the form {uid:
                {metric_name0: metric_name0_values, metric_name1: metric_name1_values}}
        """
        sample_metrics = {}
        for evaluator_label, metric_name in self.metrics.items():
            if evaluator_label not in metrics.keys():
                raise KeyError(f'Metrics not found for evaluator {evaluator_label}')
            if metric_name not in metrics[evaluator_label].keys():
                raise KeyError(f'Metric {metric_name} not found for evaluator {evaluator_label}')
            curr_metric_dict = metrics[evaluator_label][metric_name]
            if 'uid' not in curr_metric_dict.keys():
                raise KeyError(f"key 'uid' not found for metric {evaluator_label}/{metric_name}")
            n_samples = len(curr_metric_dict['uid'])
            metric_keys = set(curr_metric_dict.keys())
            metric_keys.remove('uid')
            # Check that the # of uids matches the number of metric values
            for k in metric_keys:
                v = curr_metric_dict[k]
                if len(v) != n_samples:
                    raise IndexError(
                        f'Number of samples for {k} in metric {metric_name} and evaluator {evaluator_label} is not equal to number of uids'
                    )
            # Iterate through data samples and create sample_metrics entry for each one
            for sample_n in range(n_samples):
                curr_uid = curr_metric_dict['uid'][sample_n]
                # Create entry if it does not already exist
                if curr_uid not in sample_metrics.keys():
                    sample_metrics[curr_uid] = {}
                # Iterate through metric keys
                for k in metric_keys:
                    sample_metrics[curr_uid][k] = curr_metric_dict[k][sample_n]
        return sample_metrics

    def eval_end(self, state: State, logger: Logger) -> None:
        if self.training_finished:
            if dist.get_global_rank() == 0:
                sample_metrics = self.collate_sample_metrics(state.current_metrics)
                with tempfile.TemporaryDirectory() as tmpdir:
                    file_name = os.path.join(tmpdir, self.filename)
                    with open(file_name, 'wb') as f:
                        pickle.dump(sample_metrics, f)
                    logger.file_artifact(
                        LogLevel.FIT,
                        artifact_name=f'{state.run_name}/{self.filename}',
                        file_path=f.name,
                        overwrite=True,
                    )

    def fit_end(self, state: State, logger: Logger) -> None:
        self.training_finished = True

    def save_data(data: Dict, directory: str, filename: str):
        pass
