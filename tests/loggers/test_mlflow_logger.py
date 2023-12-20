# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import csv
import json
import os
import time
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import yaml
from torch.utils.data import DataLoader

from composer.core import Callback, State
from composer.loggers import Logger, MLFlowLogger
from composer.trainer import Trainer
from tests.common.datasets import RandomImageDataset
from tests.common.markers import device
from tests.common.models import SimpleConvModel
from tests.models.test_hf_model import check_hf_model_equivalence, check_hf_tokenizer_equivalence


def _get_latest_mlflow_run(experiment_name, tracking_uri=None):
    pytest.importorskip('mlflow')
    from mlflow import MlflowClient

    # NB: Convert tracking URI to string because MlflowClient doesn't support non-string
    # (e.g. PosixPath) tracking URI representations
    client = MlflowClient(str(tracking_uri))
    experiment_id = (client.get_experiment_by_name(experiment_name).experiment_id)
    first_run_or_empty = client.search_runs(
        experiment_ids=[experiment_id],
        max_results=1,
        order_by=['start_time DESC'],
    )
    if first_run_or_empty:
        return first_run_or_empty[0]
    else:
        raise ValueError(f'Experiment with name {experiment_name} is unexpectedly empty')


def test_mlflow_experiment_init_unspecified(monkeypatch):
    """ Test that MLFlow experiment is set up correctly when no parameters are specified

    This mocks the mlflow library to check that the correct calls are made to set up the experiment
    """
    mlflow = pytest.importorskip('mlflow')
    from mlflow import MlflowClient

    monkeypatch.setattr(mlflow, 'set_tracking_uri', MagicMock())
    monkeypatch.setattr(mlflow, 'start_run', MagicMock())

    mock_state = MagicMock()
    mock_state.run_name = 'dummy-run-name'

    unspecified = MLFlowLogger()
    unspecified.init(state=mock_state, logger=MagicMock())

    assert unspecified.run_name == 'dummy-run-name'
    assert unspecified.experiment_name == 'my-mlflow-experiment'

    tracking_uri = mlflow.get_tracking_uri()
    assert MlflowClient(tracking_uri=tracking_uri).get_experiment_by_name('my-mlflow-experiment')
    assert (_get_latest_mlflow_run(
        experiment_name=unspecified.experiment_name,
        tracking_uri=tracking_uri,
    ).info.run_name == unspecified.run_name)


def test_mlflow_experiment_init_specified(monkeypatch):
    """ Test that MLFlow experiment is set up correctly when all parameters are specified

    This mocks the mlflow library to check that the correct calls are made to set up the experiment
    """
    mlflow = pytest.importorskip('mlflow')
    from mlflow import MlflowClient

    monkeypatch.setattr(mlflow, 'set_tracking_uri', MagicMock())
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

    mlflow_client = MlflowClient(tracking_uri=mlflow_uri)
    assert mlflow_client.get_experiment_by_name(specified.experiment_name)
    assert (_get_latest_mlflow_run(
        experiment_name=mlflow_exp_name,
        tracking_uri=mlflow_uri,
    ).info.run_name == specified.run_name)


def test_mlflow_experiment_init_ids(monkeypatch):
    """ Test that MLFlow experiment is set up correctly when ids in the environment are specified

    This mocks the mlflow library to check that the correct calls are made to set up the experiment
    """
    mlflow = pytest.importorskip('mlflow')

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
    assert mlflow.set_tracking_uri.call_count == 1  # We call this once in the init
    assert mlflow.set_experiment.called_with(experiment_id=mlflow_exp_id)
    assert mlflow.start_run.called_with(run_id=mlflow_run_id)


def test_mlflow_experiment_init_experiment_name(monkeypatch):
    """ Test that MLFlow experiment is set up correctly when experiment name is specified

    This mocks the mlflow library to check that the correct calls are made to set up the experiment
    """
    mlflow = pytest.importorskip('mlflow')

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

    id_logger.post_close()


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

    run = _get_latest_mlflow_run(
        experiment_name=mlflow_exp_name,
        tracking_uri=mlflow_uri,
    )
    run_id = run.info.run_id
    experiment_id = run.info.experiment_id

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


def test_mlflow_log_table(tmp_path):
    pytest.importorskip('mlflow')

    mlflow_uri = tmp_path / Path('my-test-mlflow-uri')
    mlflow_exp_name = 'test-log-table-exp-name'
    test_mlflow_logger = MLFlowLogger(
        tracking_uri=mlflow_uri,
        experiment_name=mlflow_exp_name,
    )

    mock_state = MagicMock()
    mock_state.run_name = 'dummy-run-name'  # this run name should be unused.
    mock_logger = MagicMock()

    test_mlflow_logger.init(state=mock_state, logger=mock_logger)

    run = _get_latest_mlflow_run(mlflow_exp_name, tracking_uri=mlflow_uri)
    run_info = run.info
    run_id = run_info.run_id
    experiment_id = run_info.experiment_id
    run_file_path = mlflow_uri / Path(experiment_id) / Path(run_id)

    # Create log table to test.
    columns = ['prompt', 'generation']
    rows = [['p0', 'g0'], ['p1', 'g1']]
    name = 'test_table'
    test_mlflow_logger.log_table(columns=columns, rows=rows, name='test_table')

    test_mlflow_logger.post_close()

    table_file = run_file_path / Path('artifacts') / Path(f'{name}.json')

    # Check that the table file exists.
    assert table_file.exists()

    # Check that table file contents match what was logged.
    table = json.load(open(table_file))
    assert table['columns'] == columns
    assert table['data'] == rows


@pytest.mark.filterwarnings("ignore:.*The 'transformers' MLflow Models integration.*:FutureWarning")
def test_mlflow_log_model(tmp_path, tiny_gpt2_model, tiny_gpt2_tokenizer):
    mlflow = pytest.importorskip('mlflow')

    mlflow_uri = tmp_path / Path('my-test-mlflow-uri')
    mlflow_exp_name = 'test-log-model-exp-name'
    test_mlflow_logger = MLFlowLogger(
        tracking_uri=mlflow_uri,
        experiment_name=mlflow_exp_name,
    )

    mock_state = MagicMock()
    mock_state.run_name = 'dummy-run-name'  # this run name should be unused.
    mock_logger = MagicMock()

    test_mlflow_logger.init(state=mock_state, logger=mock_logger)
    test_mlflow_logger.log_model(
        flavor='transformers',
        transformers_model={
            'model': tiny_gpt2_model,
            'tokenizer': tiny_gpt2_tokenizer,
        },
        artifact_path='my_model',
        metadata={'task': 'llm/v1/completions'},
        task='text-generation',
    )
    test_mlflow_logger.post_close()

    run = _get_latest_mlflow_run(mlflow_exp_name, tracking_uri=mlflow_uri)
    run_info = run.info
    run_id = run_info.run_id
    experiment_id = run_info.experiment_id
    run_file_path = mlflow_uri / Path(experiment_id) / Path(run_id)

    model_directory = run_file_path / Path('artifacts') / Path('my_model')
    loaded_model = mlflow.transformers.load_model(model_directory, return_type='components')

    check_hf_model_equivalence(loaded_model['model'], tiny_gpt2_model)
    check_hf_tokenizer_equivalence(loaded_model['tokenizer'], tiny_gpt2_tokenizer)


@pytest.mark.filterwarnings('ignore:.*Setuptools is replacing distutils.*:UserWarning')
@pytest.mark.filterwarnings("ignore:.*The 'transformers' MLflow Models integration.*:FutureWarning")
def test_mlflow_save_model(tmp_path, tiny_gpt2_model, tiny_gpt2_tokenizer):
    mlflow = pytest.importorskip('mlflow')

    mlflow_uri = tmp_path / Path('my-test-mlflow-uri')
    mlflow_exp_name = 'test-log-model-exp-name'
    test_mlflow_logger = MLFlowLogger(
        tracking_uri=mlflow_uri,
        experiment_name=mlflow_exp_name,
    )

    mock_state = MagicMock()
    mock_state.run_name = 'dummy-run-name'  # this run name should be unused.
    mock_logger = MagicMock()

    local_mlflow_save_path = str(tmp_path / Path('my_model_local'))
    test_mlflow_logger.init(state=mock_state, logger=mock_logger)
    test_mlflow_logger.save_model(
        flavor='transformers',
        transformers_model={
            'model': tiny_gpt2_model,
            'tokenizer': tiny_gpt2_tokenizer,
        },
        path=local_mlflow_save_path,
        metadata={'task': 'llm/v1/completions'},
        task='text-generation',
    )
    test_mlflow_logger.post_close()

    loaded_model = mlflow.transformers.load_model(local_mlflow_save_path, return_type='components')

    check_hf_model_equivalence(loaded_model['model'], tiny_gpt2_model)
    check_hf_tokenizer_equivalence(loaded_model['tokenizer'], tiny_gpt2_tokenizer)


@pytest.mark.filterwarnings('ignore:.*Setuptools is replacing distutils.*:UserWarning')
@pytest.mark.filterwarnings("ignore:.*The 'transformers' MLflow Models integration.*:FutureWarning")
def test_mlflow_register_model(tmp_path, monkeypatch):
    mlflow = pytest.importorskip('mlflow')

    monkeypatch.setattr(mlflow, 'register_model', MagicMock())

    mlflow_uri = tmp_path / Path('my-test-mlflow-uri')
    mlflow_exp_name = 'test-log-model-exp-name'
    test_mlflow_logger = MLFlowLogger(
        tracking_uri=mlflow_uri,
        experiment_name=mlflow_exp_name,
        model_registry_prefix='my_catalog.my_schema',
        model_registry_uri='databricks-uc',
    )

    mock_state = MagicMock()
    mock_state.run_name = 'dummy-run-name'  # this run name should be unused.
    mock_logger = MagicMock()

    local_mlflow_save_path = str(tmp_path / Path('my_model_local'))
    test_mlflow_logger.init(state=mock_state, logger=mock_logger)

    test_mlflow_logger.register_model(
        model_uri=local_mlflow_save_path,
        name='my_model',
    )

    assert mlflow.register_model.called_with(model_uri=local_mlflow_save_path,
                                             name='my_catalog.my_schema.my_model',
                                             await_registration_for=300,
                                             tags=None,
                                             registry_uri='databricks-uc')
    assert mlflow.get_registry_uri() == 'databricks-uc'

    test_mlflow_logger.post_close()


@pytest.mark.filterwarnings('ignore:.*Setuptools is replacing distutils.*:UserWarning')
@pytest.mark.filterwarnings("ignore:.*The 'transformers' MLflow Models integration.*:FutureWarning")
def test_mlflow_register_model_non_databricks(tmp_path, monkeypatch):
    mlflow = pytest.importorskip('mlflow')

    monkeypatch.setattr(mlflow, 'register_model', MagicMock())

    mlflow_uri = tmp_path / Path('my-test-mlflow-uri')
    mlflow_exp_name = 'test-log-model-exp-name'
    test_mlflow_logger = MLFlowLogger(
        tracking_uri=mlflow_uri,
        experiment_name=mlflow_exp_name,
        model_registry_uri='my_registry_uri',
    )

    assert mlflow.get_registry_uri() == 'my_registry_uri'

    mock_state = MagicMock()
    mock_state.run_name = 'dummy-run-name'  # this run name should be unused.
    mock_logger = MagicMock()

    local_mlflow_save_path = str(tmp_path / Path('my_model_local'))
    test_mlflow_logger.init(state=mock_state, logger=mock_logger)

    test_mlflow_logger.register_model(
        model_uri=local_mlflow_save_path,
        name='my_model',
    )

    assert mlflow.register_model.called_with(model_uri=local_mlflow_save_path,
                                             name='my_model',
                                             await_registration_for=300,
                                             tags=None,
                                             registry_uri='my_registry_uri')

    test_mlflow_logger.post_close()


@pytest.mark.filterwarnings('ignore:.*Setuptools is replacing distutils.*:UserWarning')
@pytest.mark.filterwarnings("ignore:.*The 'transformers' MLflow Models integration.*:FutureWarning")
def test_mlflow_register_uc_error(tmp_path, monkeypatch):
    mlflow = pytest.importorskip('mlflow')

    monkeypatch.setattr(mlflow, 'register_model', MagicMock())

    mlflow_uri = tmp_path / Path('my-test-mlflow-uri')
    mlflow_exp_name = 'test-log-model-exp-name'
    with pytest.raises(ValueError, match='When registering to Unity Catalog'):
        _ = MLFlowLogger(
            tracking_uri=mlflow_uri,
            experiment_name=mlflow_exp_name,
            model_registry_uri='databricks-uc',
        )


@device('cpu')
def test_mlflow_logging_works(tmp_path, device):
    mlflow = pytest.importorskip('mlflow')

    mlflow_uri = tmp_path / Path('my-test-mlflow-uri')
    experiment_name = 'mlflow_logging_test'
    test_mlflow_logger = MLFlowLogger(
        tracking_uri=mlflow_uri,
        experiment_name=experiment_name,
        log_system_metrics=True,
    )
    # Reduce the system metrics sampling interval to speed up the test.
    mlflow.set_system_metrics_sampling_interval(1)

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
    # Allow async logging to finish.
    time.sleep(3)
    test_mlflow_logger.post_close()

    run = _get_latest_mlflow_run(
        experiment_name=experiment_name,
        tracking_uri=mlflow_uri,
    )
    run_id = run.info.run_id
    experiment_id = run.info.experiment_id

    run_file_path = mlflow_uri / Path(experiment_id) / Path(run_id)

    # Test metrics logged.
    for metric_name in [
            'metrics/train/MulticlassAccuracy',
            'metrics/eval/MulticlassAccuracy',
            'metrics/eval/CrossEntropy',
            'loss/train/total',
    ]:
        metric_file = run_file_path / Path('metrics') / Path(metric_name)
        with open(metric_file) as f:
            csv_reader = csv.reader(f, delimiter=' ')
            lines = list(csv_reader)

        assert len(lines) == num_batches

    # Test params logged.
    param_path = run_file_path / Path('params')
    actual_params_list = [param_filepath.stem for param_filepath in param_path.iterdir()]

    expected_params_list = [
        'num_cpus_per_node', 'node_name', 'num_nodes', 'rank_zero_seed', 'composer_version', 'composer_commit_hash'
    ]
    assert set(expected_params_list) == set(actual_params_list)

    # Test system metrics logged.
    metric_file = run_file_path / Path('metrics') / Path('system/cpu_utilization_percentage')
    assert os.path.exists(metric_file)

    # Undo the setup to avoid affecting other test cases.
    mlflow.set_system_metrics_sampling_interval(None)


@device('cpu')
def test_mlflow_log_image_works(tmp_path, device):
    pytest.importorskip('mlflow')

    class ImageLogger(Callback):

        def before_forward(self, state: State, logger: Logger):
            inputs = state.batch_get_item(key=0)
            images = inputs.data.cpu().numpy()
            logger.log_images(images, step=state.timestamp.batch.value)
            with pytest.warns(UserWarning):
                logger.log_images(images,
                                  step=state.timestamp.batch.value,
                                  masks={'a': np.ones((2, 2))},
                                  mask_class_labels={1: 'a'})

    mlflow_uri = tmp_path / Path('my-test-mlflow-uri')
    experiment_name = 'mlflow_logging_test'
    test_mlflow_logger = MLFlowLogger(tracking_uri=mlflow_uri, experiment_name=experiment_name)

    dataset_size = 64
    batch_size = 4
    num_batches = 4
    eval_interval = '1ba'

    expected_num_ims = num_batches * batch_size

    trainer = Trainer(model=SimpleConvModel(),
                      loggers=test_mlflow_logger,
                      train_dataloader=DataLoader(RandomImageDataset(size=dataset_size), batch_size),
                      eval_dataloader=DataLoader(RandomImageDataset(size=dataset_size), batch_size),
                      max_duration=f'{num_batches}ba',
                      eval_interval=eval_interval,
                      callbacks=ImageLogger(),
                      device=device)

    trainer.fit()
    test_mlflow_logger.post_close()

    run = _get_latest_mlflow_run(
        experiment_name=experiment_name,
        tracking_uri=mlflow_uri,
    )
    run_id = run.info.run_id
    experiment_id = run.info.experiment_id

    run_file_path = mlflow_uri / Path(experiment_id) / Path(run_id)
    im_dir = run_file_path / Path('artifacts')
    assert len(os.listdir(im_dir)) == expected_num_ims
