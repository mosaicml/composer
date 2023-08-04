# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import csv
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml
from torch.utils.data import DataLoader

from composer.loggers import MLFlowLogger
from composer.trainer import Trainer
from tests.common.datasets import RandomImageDataset
from tests.common.markers import device
from tests.common.models import SimpleConvModel


def test_mlflow_experiment_init_unspecified(monkeypatch):
    """ Test that MLFlow experiment is set up correctly when no parameters are specified

    This mocks the mlflow library to check that the correct calls are made to set up the experiment
    """

    import mlflow

    monkeypatch.setattr(mlflow, 'set_tracking_uri', MagicMock())
    monkeypatch.setattr(mlflow, 'set_experiment', MagicMock())
    monkeypatch.setattr(mlflow, 'start_run', MagicMock())

    mock_state = MagicMock()
    mock_state.run_name = 'dummy-run-name'

    unspecified = MLFlowLogger()
    unspecified.init(state=mock_state, logger=MagicMock())

    assert unspecified.run_name == 'dummy-run-name'
    assert unspecified.experiment_name == 'my-mlflow-experiment'
    assert mlflow.set_tracking_uri.call_count == 0
    assert mlflow.set_experiment.called_with(experiment_name='my-mlflow-experiment')
    assert mlflow.start_run.called_with(run_name='dummy-run-name')


def test_mlflow_experiment_init_specified(monkeypatch):
    """ Test that MLFlow experiment is set up correctly when all parameters are specified

    This mocks the mlflow library to check that the correct calls are made to set up the experiment
    """
    import mlflow

    monkeypatch.setattr(mlflow, 'set_tracking_uri', MagicMock())
    monkeypatch.setattr(mlflow, 'set_experiment', MagicMock())
    monkeypatch.setattr(mlflow, 'start_run', MagicMock())

    mock_state = MagicMock()
    mock_state.run_name = 'dummy-run-name'  # Not used

    mlflow_uri = 'my-test-mlflow-uri'
    mlflow_exp_name = 'my-test-mlflow-exp'
    mlflow_run_name = 'my-test-mlflow-run'

    specified = MLFlowLogger(
        experiment_name=mlflow_exp_name,
        run_name=mlflow_run_name,
        tracking_uri=mlflow_uri,
        rank_zero_only=False,
    )
    specified.init(state=mock_state, logger=MagicMock())

    exp_run_name = 'my-test-mlflow-run-rank0'

    assert specified.run_name == exp_run_name
    assert specified.experiment_name == mlflow_exp_name
    assert mlflow.set_tracking_uri.call_count == 1
    assert mlflow.set_tracking_uri.called_with(mlflow_uri)
    assert mlflow.set_experiment.called_with(experiment_name=mlflow_exp_name)
    assert mlflow.start_run.called_with(run_name=exp_run_name)


def test_mlflow_experiment_init_ids(monkeypatch):
    """ Test that MLFlow experiment is set up correctly when ids in the environment are specified

    This mocks the mlflow library to check that the correct calls are made to set up the experiment
    """
    import mlflow

    monkeypatch.setattr(mlflow, 'set_tracking_uri', MagicMock())
    monkeypatch.setattr(mlflow, 'set_experiment', MagicMock())
    monkeypatch.setattr(mlflow, 'start_run', MagicMock())

    mock_state = MagicMock()
    mock_state.run_name = 'dummy-run-name'  # Not used

    mlflow_exp_id = '123'
    mlflow_run_id = '456'

    monkeypatch.setenv(mlflow.environment_variables.MLFLOW_RUN_ID.name, mlflow_run_id)
    monkeypatch.setenv(mlflow.environment_variables.MLFLOW_EXPERIMENT_ID.name, mlflow_exp_id)

    id_logger = MLFlowLogger()
    id_logger.init(state=mock_state, logger=MagicMock())

    assert id_logger.run_name == 'dummy-run-name'  # Defaults are set, but we don't use them
    assert id_logger.experiment_name == 'my-mlflow-experiment'
    assert mlflow.set_tracking_uri.call_count == 0
    assert mlflow.set_experiment.called_with(experiment_id=mlflow_exp_id)
    assert mlflow.start_run.called_with(run_id=mlflow_run_id)


def test_mlflow_experiment_init_experiment_name(monkeypatch):
    """ Test that MLFlow experiment is set up correctly when experiment name is specified

    This mocks the mlflow library to check that the correct calls are made to set up the experiment
    """
    import mlflow

    monkeypatch.setattr(mlflow, 'set_tracking_uri', MagicMock())
    monkeypatch.setattr(mlflow, 'set_experiment', MagicMock())
    monkeypatch.setattr(mlflow, 'start_run', MagicMock())

    mock_state = MagicMock()
    mock_state.run_name = 'dummy-run-name'

    exp_name = 'foobar'
    monkeypatch.setenv(mlflow.environment_variables.MLFLOW_EXPERIMENT_NAME.name, exp_name)

    id_logger = MLFlowLogger()
    id_logger.init(state=mock_state, logger=MagicMock())

    assert id_logger.experiment_name == exp_name
    assert mlflow.set_experiment.called_with(experiment_name=exp_name)


def test_mlflow_experiment_set_up(tmp_path):
    """ Test that MLFlow experiment is set up correctly within mlflow
    """
    mlflow = pytest.importorskip('mlflow')

    mlflow_uri = tmp_path / Path('my-test-mlflow-uri')
    mlflow_exp_name = 'my-test-mlflow-exp'
    mlflow_run_name = 'my-test-mlflow-run'

    test_mlflow_logger = MLFlowLogger(experiment_name=mlflow_exp_name,
                                      run_name=mlflow_run_name,
                                      tracking_uri=mlflow_uri)

    mock_state = MagicMock()
    mock_state.run_name = 'dummy-run-name'  # this run name should be unused.
    mock_logger = MagicMock()

    test_mlflow_logger.init(state=mock_state, logger=mock_logger)

    run_info = mlflow.active_run().info
    run_id = run_info.run_id
    experiment_id = run_info.experiment_id

    # Check uri set correctly.
    assert mlflow_uri.exists()

    # Check experiment name set correctly.
    exp_cfg_file_path = mlflow_uri / Path(experiment_id) / Path('meta.yaml')
    exp_cfg = yaml.safe_load(open(str(exp_cfg_file_path), 'r'))
    expected_exp_name = mlflow_exp_name
    actual_exp_name = exp_cfg['name']
    assert actual_exp_name == expected_exp_name

    # Check run_name set correctly.
    run_cfg_file_path = mlflow_uri / Path(experiment_id) / Path(run_id) / Path('meta.yaml')
    run_cfg = yaml.safe_load(open(str(run_cfg_file_path), 'r'))
    expected_run_name = mlflow_run_name
    actual_run_name = run_cfg['run_name']
    assert actual_run_name == expected_run_name

    # Check run ended.
    test_mlflow_logger.post_close()
    assert mlflow.active_run() is None

    # Check new run can be created.
    del test_mlflow_logger
    test_mlflow_logger = MLFlowLogger(experiment_name=mlflow_exp_name, run_name=mlflow_run_name + '_new')
    test_mlflow_logger.init(state=mock_state, logger=mock_logger)
    test_mlflow_logger.post_close()


@device('cpu')
def test_mlflow_logging_works(tmp_path, device):
    mlflow = pytest.importorskip('mlflow')
    mlflow_uri = tmp_path / Path('my-test-mlflow-uri')
    test_mlflow_logger = MLFlowLogger(tracking_uri=mlflow_uri)

    dataset_size = 64
    batch_size = 4
    num_batches = 4
    eval_interval = '1ba'

    trainer = Trainer(model=SimpleConvModel(),
                      loggers=test_mlflow_logger,
                      train_dataloader=DataLoader(RandomImageDataset(size=dataset_size), batch_size),
                      eval_dataloader=DataLoader(RandomImageDataset(size=dataset_size), batch_size),
                      max_duration=f'{num_batches}ba',
                      eval_interval=eval_interval,
                      device=device)
    trainer.fit()

    run_info = mlflow.active_run().info
    run_id = run_info.run_id
    experiment_id = run_info.experiment_id

    run_file_path = mlflow_uri / Path(experiment_id) / Path(run_id)

    # Test metrics logged.
    for metric_name in [
            'metrics/train/MulticlassAccuracy', 'metrics/eval/MulticlassAccuracy', 'metrics/eval/CrossEntropy',
            'loss/train/total'
    ]:
        metric_file = run_file_path / Path('metrics') / Path(metric_name)
        with open(metric_file) as f:
            csv_reader = csv.reader(f, delimiter=' ')
            lines = [line for line in csv_reader]

        assert len(lines) == num_batches

    # Test params logged.
    param_path = run_file_path / Path('params')
    actual_params_list = [param_filepath.stem for param_filepath in param_path.iterdir()]

    expected_params_list = ['num_cpus_per_node', 'node_name', 'num_nodes', 'rank_zero_seed']
    assert set(expected_params_list) == set(actual_params_list)
