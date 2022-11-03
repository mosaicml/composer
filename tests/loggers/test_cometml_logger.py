# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0
import os
import zipfile
from collections import defaultdict
from json import JSONDecoder
from pathlib import Path

import pytest
from torch.utils.data import DataLoader

from composer.trainer import Trainer
from tests.common import RandomClassificationDataset, SimpleModel


def test_comet_ml_logging_train_loop(monkeypatch, tmp_path):
    pytest.importorskip('comet_ml', reason='comet_ml is optional')
    import comet_ml

    monkeypatch.setattr(comet_ml, 'Experiment', comet_ml.OfflineExperiment)
    from composer.loggers import CometMLLogger

    # Set offline directory.
    offline_directory = str(tmp_path / Path('.my_cometml_runs'))
    os.environ['COMET_OFFLINE_DIRECTORY'] = offline_directory

    comet_logger = CometMLLogger()

    trainer = Trainer(
        model=SimpleModel(),
        train_dataloader=DataLoader(RandomClassificationDataset()),
        train_subset_num_batches=2,
        max_duration='2ep',
        loggers=comet_logger,
    )
    trainer.fit()

    run_name = trainer.state.run_name

    del trainer

    assert comet_logger.experiment is not None
    assert comet_logger.experiment.ended

    # Open, decompress, decode, and extract offline dump of metrics.
    comet_exp_dump_filepath = str(Path(offline_directory) / Path(comet_logger.experiment.id).with_suffix('.zip'))
    zf = zipfile.ZipFile(comet_exp_dump_filepath)
    comet_logs_path = zf.extract('messages.json', path=offline_directory)
    jd = JSONDecoder()
    msg_type_to_msgs = defaultdict(list)

    with open(comet_logs_path) as f:
        for line in f.readlines():
            parsed_line = jd.decode(line)
            msg_type_to_msgs[parsed_line['type']].append(parsed_line['payload'])

    # Check that init set the run name
    assert comet_logger.name == run_name
    assert comet_logger.experiment.name == run_name

    # Check that basic metrics appear in the comet logs
    assert len([
        metric_msg for metric_msg in msg_type_to_msgs['metric_msg'] if metric_msg['metric']['metricName'] == 'epoch'
    ]) == 2

    # Check that basic params appear in the comet logs
    assert len([
        param_msg for param_msg in msg_type_to_msgs['parameter_msg']
        if param_msg['param']['paramName'] == 'rank_zero_seed'
    ]) > 0


def test_comet_ml_post_close(monkeypatch, tmp_path):
    pytest.importorskip('comet_ml', reason='comet_ml is optional')
    import comet_ml

    monkeypatch.setattr(comet_ml, 'Experiment', comet_ml.OfflineExperiment)
    from composer.loggers import CometMLLogger

    # Set offline directory.
    offline_directory = str(tmp_path / Path('.my_cometml_runs'))
    os.environ['COMET_OFFLINE_DIRECTORY'] = offline_directory

    comet_logger = CometMLLogger()
    comet_logger.post_close()

    assert comet_logger.experiment is not None
    assert comet_logger.experiment.ended


def test_comet_ml_log_created_from_key(monkeypatch, tmp_path):
    pytest.importorskip('comet_ml', reason='comet_ml is optional')
    import comet_ml

    monkeypatch.setattr(comet_ml, 'Experiment', comet_ml.OfflineExperiment)
    from composer.loggers import CometMLLogger

    # Set offline directory.
    offline_directory = str(tmp_path / Path('.my_cometml_runs'))
    os.environ['COMET_OFFLINE_DIRECTORY'] = offline_directory

    comet_logger = CometMLLogger()
    comet_logger.post_close()

    assert comet_logger.experiment is not None

    # Open, decompress, decode, and check for Created from key logged
    comet_exp_dump_filepath = str(Path(offline_directory) / Path(comet_logger.experiment.id).with_suffix('.zip'))
    zf = zipfile.ZipFile(comet_exp_dump_filepath)
    comet_logs_path = zf.extract('messages.json', path=offline_directory)
    jd = JSONDecoder()
    created_from_found = False
    expected_created_from_log = {'key': 'Created from', 'val': 'mosaicml-composer'}
    with open(comet_logs_path) as f:
        for line in f.readlines():
            comet_msg = jd.decode(line)
            if comet_msg['type'] == 'ws_msg' and comet_msg['payload'].get('log_other', {}) == expected_created_from_log:
                created_from_found = True

    assert created_from_found


def test_comet_ml_log_metrics_and_hyperparameters(monkeypatch, tmp_path):
    """Check metrics logged with CometMLLogger are properly written to offline dump."""
    pytest.importorskip('comet_ml', reason='comet_ml is optional')
    import comet_ml

    # Set some dummy log values.
    steps = [0, 1, 2]
    metric_values = [0.1, 0.4, 0.7]
    metric_name = 'my_test_metric'
    param_names = ['my_cool_parameter1', 'my_cool_parameter2']
    param_values = [10, 3]

    # Set offline directory.
    offline_directory = str(tmp_path / Path('.my_cometml_runs'))
    os.environ['COMET_OFFLINE_DIRECTORY'] = offline_directory

    # Monkeypatch Experiment with OfflineExperiment to avoid uploading to CometML and
    # avoid needing an API+KEY.
    monkeypatch.setattr(comet_ml, 'Experiment', comet_ml.OfflineExperiment)
    from composer.loggers import CometMLLogger

    # Log dummy values with CometMLLogger.
    comet_logger = CometMLLogger()
    comet_logger.log_hyperparameters(dict(zip(param_names, param_values)))
    for step, metric_value in zip(steps, metric_values):
        comet_logger.log_metrics({'my_test_metric': metric_value}, step=step)

    # Simulate the post_close call to end the CometML experiment
    comet_logger.post_close()

    assert comet_logger.experiment is not None

    # Open, decompress, decode, and extract offline dump of metrics.
    comet_exp_dump_filepath = str(Path(offline_directory) / Path(comet_logger.experiment.id).with_suffix('.zip'))
    zf = zipfile.ZipFile(comet_exp_dump_filepath)
    comet_logs_path = zf.extract('messages.json', path=offline_directory)
    jd = JSONDecoder()
    metric_msgs = []
    param_msgs = []
    with open(comet_logs_path) as f:
        for line in f.readlines():
            comet_msg = jd.decode(line)
            if (comet_msg['type'] == 'metric_msg') and (comet_msg['payload']['metric']['metricName']
                                                        == 'my_test_metric'):
                metric_msgs.append(comet_msg['payload']['metric'])
            if comet_msg['type'] == 'parameter_msg' and (
                    comet_msg['payload']['param']['paramName'].startswith('my_cool')):
                param_msgs.append(comet_msg['payload']['param'])

    # Assert dummy metrics input to log_metrics are the same as
    # those written to offline dump.
    assert [msg['metricValue'] for msg in metric_msgs] == metric_values
    assert [msg['step'] for msg in metric_msgs] == steps
    assert all([msg['metricName'] == metric_name for msg in metric_msgs])

    # Assert dummy params input to log_hyperparameters are the same as
    # those written to offline dump
    assert [msg['paramValue'] for msg in param_msgs] == param_values
    assert [msg['paramName'] for msg in param_msgs] == param_names
