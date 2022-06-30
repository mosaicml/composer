# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

from typing import Iterable, Tuple
from unittest.mock import MagicMock

import pytest

from composer.core import Callback, State
from composer.loggers import Logger
from composer.trainer import Trainer
from tests.common import SimpleModel


class MetricsCallback(Callback):

    def __init__(self, compute_training_metrics: bool, compute_val_metrics: bool) -> None:
        self.compute_training_metrics = compute_training_metrics
        self.compute_val_metrics = compute_val_metrics
        self._train_batch_end_train_accuracy = None

    def init(self, state: State, logger: Logger) -> None:
        # on init, the `current_metrics` should be empty
        del logger  # unused
        assert state.current_metrics == {}, 'no metrics should be defined on init()'

    def batch_end(self, state: State, logger: Logger) -> None:
        # The metric should be computed and updated on state every batch.
        del logger  # unused
        if self.compute_training_metrics:
            # assuming that at least one sample was correctly classified
            assert state.current_metrics['train']['Accuracy'] != 0.0
            self._train_batch_end_train_accuracy = state.current_metrics['train']['Accuracy']

    def epoch_end(self, state: State, logger: Logger) -> None:
        # The metric at epoch end should be the same as on batch end.
        del logger  # unused
        if self.compute_training_metrics:
            assert state.current_metrics['train']['Accuracy'] == self._train_batch_end_train_accuracy

    def eval_end(self, state: State, logger: Logger) -> None:
        if self.compute_val_metrics:
            # assuming that at least one sample was correctly classified
            assert state.current_metrics['eval']['Accuracy'] != 0.0


@pytest.mark.parametrize('compute_training_metrics', [True, False])
@pytest.mark.parametrize('eval_interval', ['1ba', '1ep', '0ep'])
def test_current_metrics(
    dummy_train_dataloader: Iterable,
    dummy_val_dataloader: Iterable,
    dummy_num_classes: int,
    dummy_in_shape: Tuple[int, ...],
    compute_training_metrics: bool,
    eval_interval: str,
):
    # Configure the trainer
    num_channels = dummy_in_shape[0]
    mock_logger_destination = MagicMock()
    model = SimpleModel(num_features=num_channels, num_classes=dummy_num_classes)
    compute_val_metrics = eval_interval != '0ep'
    train_subset_num_batches = 2
    eval_subset_num_batches = 2
    num_epochs = 2
    metrics_callback = MetricsCallback(
        compute_training_metrics=compute_training_metrics,
        compute_val_metrics=compute_val_metrics,
    )

    # Create the trainer
    trainer = Trainer(
        model=model,
        train_dataloader=dummy_train_dataloader,
        eval_dataloader=dummy_val_dataloader,
        max_duration=num_epochs,
        compute_training_metrics=compute_training_metrics,
        train_subset_num_batches=train_subset_num_batches,
        eval_subset_num_batches=eval_subset_num_batches,
        loggers=[mock_logger_destination],
        callbacks=[metrics_callback],
        eval_interval=eval_interval,
    )

    # Train the model
    trainer.fit()

    if not compute_training_metrics and not compute_val_metrics:
        return

    # Validate the metrics
    if compute_training_metrics:
        assert trainer.state.current_metrics['train']['Accuracy'] != 0.0
    else:
        assert 'train' not in trainer.state.current_metrics

    if compute_val_metrics:
        assert trainer.state.current_metrics['eval']['Accuracy'] != 0.0
    else:
        assert 'eval' not in trainer.state.current_metrics

    # Validate that the logger was called the correct number of times for metric calls
    num_expected_calls = 0
    if compute_training_metrics:
        # computed once per batch
        # and again at epoch end
        num_expected_calls += (train_subset_num_batches + 1) * num_epochs
    # computed at eval end
    if compute_val_metrics:
        num_calls_per_eval = 1
        num_evals = 0
        if eval_interval == '1ba':
            num_evals += train_subset_num_batches * num_epochs
        if eval_interval == '1ep':
            num_evals += num_epochs
        num_expected_calls += (num_calls_per_eval) * num_evals
    num_actual_calls = 0

    # Need to filter out non-metrics-related calls
    for call in mock_logger_destination.log_data.mock_calls:
        data = call[1][2]
        for k in data:
            if k.startswith('metrics/'):
                num_actual_calls += 1
                break

    assert num_actual_calls == num_expected_calls
