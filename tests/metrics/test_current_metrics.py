# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock

import pytest
from torch.utils.data import DataLoader

from composer.core import Callback, State
from composer.loggers import Logger
from composer.trainer import Trainer
from tests.common import SimpleModel
from tests.common.datasets import RandomClassificationDataset


class MetricsCallback(Callback):

    def __init__(self, compute_val_metrics: bool) -> None:
        self.compute_val_metrics = compute_val_metrics
        self._train_batch_end_train_accuracy = None

    def init(self, state: State, logger: Logger) -> None:
        # on init, the current metrics should be empty
        del logger  # unused
        assert state.train_metrics == {}, 'no train metrics should be defined on init()'
        assert state.eval_metrics == {}, 'no eval metrics should be defined on init()'

    def batch_end(self, state: State, logger: Logger) -> None:
        # The metric should be computed and updated on state every batch.
        del logger  # unused
        # assuming that at least one sample was correctly classified
        assert state.train_metrics['Accuracy'].compute() != 0.0
        self._train_batch_end_train_accuracy = state.train_metrics['Accuracy']

    def epoch_end(self, state: State, logger: Logger) -> None:
        # The metric at epoch end should be the same as on batch end.
        del logger  # unused
        assert state.train_metrics['Accuracy'].compute() == self._train_batch_end_train_accuracy

    def eval_end(self, state: State, logger: Logger) -> None:
        if self.compute_val_metrics:
            # assuming that at least one sample was correctly classified
            assert state.eval_metrics['eval']['Accuracy'].compute() != 0.0


@pytest.mark.parametrize('eval_interval', ['1ba', '1ep', '0ep'])
def test_current_metrics(eval_interval: str,):
    # Configure the trainer
    mock_logger_destination = MagicMock()
    mock_logger_destination.log_metrics = MagicMock()
    model = SimpleModel(num_features=1, num_classes=2)
    compute_val_metrics = True if eval_interval != '0ep' else False
    train_subset_num_batches = 1
    eval_subset_num_batches = 1
    num_epochs = 1
    metrics_callback = MetricsCallback(compute_val_metrics=compute_val_metrics,)

    dataset_kwargs = {
        'num_classes': 2,
        'shape': (1, 5, 5),
    }
    # Create the trainer
    trainer = Trainer(
        model=model,
        train_dataloader=DataLoader(
            RandomClassificationDataset(**dataset_kwargs),
            batch_size=16,
        ),
        eval_dataloader=DataLoader(
            RandomClassificationDataset(**dataset_kwargs),
            batch_size=8,
        ),
        max_duration=num_epochs,
        train_subset_num_batches=train_subset_num_batches,
        eval_subset_num_batches=eval_subset_num_batches,
        loggers=[mock_logger_destination],
        callbacks=[metrics_callback],
        eval_interval=eval_interval,
    )

    # Train the model
    trainer.fit()

    if not compute_val_metrics:
        return

    # Validate the metrics
    assert trainer.state.train_metrics['Accuracy'].compute() != 0.0

    if compute_val_metrics:
        assert trainer.state.eval_metrics['eval']['Accuracy'].compute() != 0.0
    else:
        assert 'eval' not in trainer.state.eval_metrics

    num_log_step_and_index_calls_per_batch = 1  # global_step and batch_idx calls
    num_log_grad_accum_calls_per_batch = 1
    num_log_loss_calls_per_batch = 1
    num_log_epoch_calls_per_batch = 1
    num_log_train_metric_calls_per_batch = 1


    total_num_train_batches = num_epochs * train_subset_num_batches
    # Validate that the logger was called the correct number of times for metric calls

    # Every epoch is logged.
    num_log_epoch_calls = num_log_epoch_calls_per_batch * total_num_train_batches
    num_log_train_step_and_index_calls = total_num_train_batches * num_log_step_and_index_calls_per_batch
    num_log_loss_calls = total_num_train_batches * num_log_loss_calls_per_batch
    num_log_grad_accum_calls = total_num_train_batches * num_log_grad_accum_calls_per_batch
    num_log_train_metric_calls = total_num_train_batches * num_log_train_metric_calls_per_batch
    num_expected_train_log_calls = (num_log_epoch_calls + num_log_train_step_and_index_calls + num_log_loss_calls 
                                        + num_log_grad_accum_calls + num_log_train_metric_calls)

    num_expected_eval_log_calls = 0
    # computed at eval end
    if compute_val_metrics:
        num_log_calls_per_eval = 4  # eval metrics + train_metrics + epoch + trainer/global_step
        num_evals = 0
        if eval_interval == '1ba':
            num_evals = total_num_train_batches
        if eval_interval == '1ep':
            num_evals = num_epochs
        num_expected_eval_log_calls = (num_log_calls_per_eval) * num_evals

    num_expected_calls = num_expected_train_log_calls + num_expected_eval_log_calls
    num_actual_calls = len(mock_logger_destination.log_metrics.mock_calls)

    assert num_actual_calls == num_expected_calls
