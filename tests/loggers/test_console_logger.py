# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import math
import re
from pathlib import Path

import pytest
from torch.utils.data import DataLoader
from torchmetrics import MetricCollection

from composer.trainer import Trainer
from tests.common import RandomClassificationDataset, SimpleModel


@pytest.fixture
def console_logger_test_file_path(tmp_path) -> str:
    return str(Path(tmp_path) / Path('console_test'))


@pytest.fixture
def console_logger_test_stream(console_logger_test_file_path):
    return open(console_logger_test_file_path, 'w')


@pytest.mark.parametrize('log_interval_unit', ['ba', 'ep'])
@pytest.mark.parametrize('max_duration_unit', ['ba', 'ep'])
@pytest.mark.parametrize('log_interval', [1, 2, 3])
@pytest.mark.parametrize('max_duration', [8, 9, 10, 11])
def test_console_logger_interval(console_logger_test_stream, console_logger_test_file_path, log_interval, max_duration,
                                 log_interval_unit, max_duration_unit):

    batch_size = 4
    dataset_size = 17
    batches_per_epoch = math.ceil(dataset_size / batch_size)

    model = SimpleModel()
    trainer = Trainer(model=model,
                      console_stream=console_logger_test_stream,
                      console_log_interval=f'{log_interval}{log_interval_unit}',
                      log_to_console=True,
                      progress_bar=False,
                      train_dataloader=DataLoader(RandomClassificationDataset(size=dataset_size),
                                                  batch_size=batch_size),
                      max_duration=f'{max_duration}{max_duration_unit}')
    trainer.fit()
    console_logger_test_stream.flush()
    console_logger_test_stream.close()

    with open(console_logger_test_file_path, 'r') as f:
        lines = f.readlines()

    # Make a regular expression for matches for any line that contains "Train" followed by
    # a colon.
    reg_exp = re.compile('Train *:*')
    actual_num_log_lines = sum([1 if bool(reg_exp.search(line)) else 0 for line in lines])

    assert model.train_metrics is not None
    num_metrics = len(list(model.train_metrics.keys())) if isinstance(model.train_metrics, MetricCollection) else 1
    num_losses = 1
    num_metrics_and_losses_per_logging_event = num_metrics + num_losses

    if log_interval_unit == max_duration_unit:
        expected_num_logging_events = max_duration // log_interval
    elif log_interval_unit == 'ba' and max_duration_unit == 'ep':
        expected_num_logging_events = (batches_per_epoch * max_duration) // log_interval
    else:  # for the case where log_interval_unit == 'ep' and max_duration == 'ba'.
        total_epochs = max_duration // batches_per_epoch
        expected_num_logging_events = total_epochs // log_interval
    if log_interval != 1:
        expected_num_logging_events += 1  # Because we automatically log the first batch or epoch.

    expected_num_lines = expected_num_logging_events * num_metrics_and_losses_per_logging_event

    assert actual_num_log_lines == expected_num_lines


@pytest.mark.parametrize('eval_interval_unit', ['ba', 'ep'])
@pytest.mark.parametrize('max_duration_unit', ['ba', 'ep'])
@pytest.mark.parametrize('eval_interval', [2, 3])
@pytest.mark.parametrize('max_duration', [8, 9])
def test_console_logger_interval_with_eval(console_logger_test_stream, console_logger_test_file_path, eval_interval,
                                           max_duration, eval_interval_unit, max_duration_unit):

    batch_size = 4
    dataset_size = 17
    batches_per_epoch = math.ceil(dataset_size / batch_size)

    model = SimpleModel()
    trainer = Trainer(model=model,
                      console_stream=console_logger_test_stream,
                      eval_interval=f'{eval_interval}{eval_interval_unit}',
                      log_to_console=True,
                      progress_bar=False,
                      train_dataloader=DataLoader(RandomClassificationDataset(size=dataset_size),
                                                  batch_size=batch_size),
                      eval_dataloader=DataLoader(RandomClassificationDataset(size=dataset_size), batch_size=batch_size),
                      max_duration=f'{max_duration}{max_duration_unit}')
    trainer.fit()
    console_logger_test_stream.flush()
    console_logger_test_stream.close()

    with open(console_logger_test_file_path, 'r') as f:
        lines = f.readlines()

    # Make a regular expression for matches for any line that contains "Eval" followed by
    # a colon.
    eval_reg_exp = re.compile('Eval *:*')
    actual_num_eval_log_lines = sum([1 if bool(eval_reg_exp.search(line)) else 0 for line in lines])

    assert model.val_metrics is not None
    num_eval_metrics = len(list(model.val_metrics.keys())) if isinstance(model.val_metrics, MetricCollection) else 1
    num_eval_losses = 0
    num_eval_metrics_and_losses_per_logging_event = num_eval_metrics + num_eval_losses

    if eval_interval_unit == max_duration_unit:
        expected_num_eval_logging_events, remainder = divmod(max_duration, eval_interval)
    elif eval_interval_unit == 'ba' and max_duration_unit == 'ep':
        expected_num_eval_logging_events, remainder = divmod((batches_per_epoch * max_duration), eval_interval)
    else:  # for the case where eval_interval_unit == 'ep' and max_duration == 'ba'.
        batches_per_logging_event = batches_per_epoch * eval_interval
        expected_num_eval_logging_events, remainder = divmod(max_duration, batches_per_logging_event)

    # An eval logging event always happens at fit_end, so if one would not normally fall at
    # last batch or epoch, then add an extra event to the expected.
    if remainder:
        expected_num_eval_logging_events += 1

    expected_num_eval_lines = expected_num_eval_logging_events * num_eval_metrics_and_losses_per_logging_event

    assert actual_num_eval_log_lines == expected_num_eval_lines


def test_log_to_console_and_progress_bar_warning():
    with pytest.warns(Warning):
        Trainer(model=SimpleModel(),
                log_to_console=True,
                progress_bar=True,
                train_dataloader=DataLoader(RandomClassificationDataset()),
                max_duration=f'2ep')
