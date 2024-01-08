# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from composer.utils import MLFlowObjectStore
from composer.utils.object_store.mlflow_object_store import PLACEHOLDER_EXPERIMENT_ID, PLACEHOLDER_RUN_ID

TEST_PATH_FORMAT = 'databricks/mlflow-tracking/{experiment_id}/{run_id}/artifacts/'
EXPERIMENT_ID = '123'
EXPERIMENT_NAME = 'test-experiment'
RUN_ID = '456'
RUN_NAME = 'test-run'
ARTIFACT_PATH = 'path/to/artifact'
DEFAULT_PATH = TEST_PATH_FORMAT.format(experiment_id=EXPERIMENT_ID, run_id=RUN_ID)


def test_parse_dbfs_path():
    full_artifact_path = DEFAULT_PATH + ARTIFACT_PATH
    assert MLFlowObjectStore.parse_dbfs_path(full_artifact_path) == (EXPERIMENT_ID, RUN_ID, ARTIFACT_PATH)

    # Test with bad prefix
    with pytest.raises(ValueError):
        MLFlowObjectStore.parse_dbfs_path(f'bad-prefix/{EXPERIMENT_ID}/{RUN_ID}/artifacts/{ARTIFACT_PATH}')

    # Test without artifacts
    with pytest.raises(ValueError):
        MLFlowObjectStore.parse_dbfs_path(f'databricks/mlflow-tracking/{EXPERIMENT_ID}/{RUN_ID}/')
    with pytest.raises(ValueError):
        MLFlowObjectStore.parse_dbfs_path(f'databricks/mlflow-tracking/{EXPERIMENT_ID}/{RUN_ID}/not-artifacts/')


def test_init_fail_without_databricks_tracking_uri(monkeypatch):
    monkeypatch.setenv('MLFLOW_TRACKING_URI', 'not-databricks')
    with pytest.raises(ValueError):
        MLFlowObjectStore(DEFAULT_PATH)


def test_init_with_experiment_and_run(monkeypatch):
    dbx_sdk = pytest.importorskip('databricks.sdk')
    monkeypatch.setattr(dbx_sdk, 'WorkspaceClient', MagicMock())

    mlflow = pytest.importorskip('mlflow')
    mock_mlflow_client = MagicMock()
    monkeypatch.setattr(mlflow, 'MlflowClient', mock_mlflow_client)

    mock_mlflow_client.return_value.get_run.return_value = MagicMock(info=MagicMock(experiment_id=EXPERIMENT_ID))

    store = MLFlowObjectStore(DEFAULT_PATH)
    assert store.experiment_id == EXPERIMENT_ID
    assert store.run_id == RUN_ID


def test_init_with_experiment_and_no_run(monkeypatch):
    dbx_sdk = pytest.importorskip('databricks.sdk')
    monkeypatch.setattr(dbx_sdk, 'WorkspaceClient', MagicMock())

    mlflow = pytest.importorskip('mlflow')
    mock_mlflow_client = MagicMock()
    monkeypatch.setattr(mlflow, 'MlflowClient', mock_mlflow_client)

    mock_mlflow_client.return_value.create_run.return_value = MagicMock(
        info=MagicMock(run_id=RUN_ID, run_name='test-run'))

    store = MLFlowObjectStore(TEST_PATH_FORMAT.format(experiment_id=EXPERIMENT_ID, run_id=PLACEHOLDER_RUN_ID))
    assert store.experiment_id == EXPERIMENT_ID
    assert store.run_id == RUN_ID


def test_init_with_run_and_no_experiment(monkeypatch):
    dbx_sdk = pytest.importorskip('databricks.sdk')
    monkeypatch.setattr(dbx_sdk, 'WorkspaceClient', MagicMock())

    with pytest.raises(ValueError):
        MLFlowObjectStore(TEST_PATH_FORMAT.format(experiment_id=PLACEHOLDER_EXPERIMENT_ID, run_id=RUN_ID))


def test_init_with_active_run(monkeypatch):
    dbx_sdk = pytest.importorskip('databricks.sdk')
    monkeypatch.setattr(dbx_sdk, 'WorkspaceClient', MagicMock())

    mlflow = pytest.importorskip('mlflow')
    mock_active_run = MagicMock()
    monkeypatch.setattr(mlflow, 'active_run', mock_active_run)
    monkeypatch.setattr(mlflow, 'MlflowClient', MagicMock())

    mock_active_run.return_value = MagicMock(info=MagicMock(experiment_id=EXPERIMENT_ID, run_id=RUN_ID))

    store = MLFlowObjectStore(
        TEST_PATH_FORMAT.format(experiment_id=PLACEHOLDER_EXPERIMENT_ID, run_id=PLACEHOLDER_RUN_ID))
    assert store.experiment_id == EXPERIMENT_ID
    assert store.run_id == RUN_ID


def test_init_with_existing_experiment_and_no_run(monkeypatch):
    dbx_sdk = pytest.importorskip('databricks.sdk')
    monkeypatch.setattr(dbx_sdk, 'WorkspaceClient', MagicMock())

    mlflow = pytest.importorskip('mlflow')
    mock_mlflow_client = MagicMock()
    monkeypatch.setattr(mlflow, 'MlflowClient', mock_mlflow_client)

    mock_mlflow_client.return_value.get_experiment_by_name.return_value = MagicMock(experiment_id=EXPERIMENT_ID)
    mock_mlflow_client.return_value.create_run.return_value = MagicMock(
        info=MagicMock(run_id=RUN_ID, run_name='test-run'))

    store = MLFlowObjectStore(
        TEST_PATH_FORMAT.format(experiment_id=PLACEHOLDER_EXPERIMENT_ID, run_id=PLACEHOLDER_RUN_ID))
    assert store.experiment_id == EXPERIMENT_ID
    assert store.run_id == RUN_ID


def test_init_with_no_experiment_and_no_run(monkeypatch):
    dbx_sdk = pytest.importorskip('databricks.sdk')
    monkeypatch.setattr(dbx_sdk, 'WorkspaceClient', MagicMock())

    mlflow = pytest.importorskip('mlflow')
    mock_mlflow_client = MagicMock()
    monkeypatch.setattr(mlflow, 'MlflowClient', mock_mlflow_client)

    mock_mlflow_client.return_value.get_experiment_by_name.return_value = None
    mock_mlflow_client.return_value.create_experiment.return_value = EXPERIMENT_ID
    mock_mlflow_client.return_value.create_run.return_value = MagicMock(
        info=MagicMock(run_id=RUN_ID, run_name='test-run'))

    store = MLFlowObjectStore(
        TEST_PATH_FORMAT.format(experiment_id=PLACEHOLDER_EXPERIMENT_ID, run_id=PLACEHOLDER_RUN_ID))
    assert store.experiment_id == EXPERIMENT_ID
    assert store.run_id == RUN_ID


@pytest.fixture()
def mlflow_object_store(monkeypatch):

    def mock_mlflow_client_list_artifacts(*args, **kwargs):
        """Mock behavior for MlflowClient.list_artifacts().

        Behaves as if artifacts are stored under the following structure:
        - dir1/
            - a.txt
            - b.txt
        - dir2/
            - c.txt
            - dir3/
                - d.txt
        """
        path = args[1]
        if not path:
            return [
                MagicMock(path='dir1', is_dir=True, file_size=None),
                MagicMock(path='dir2', is_dir=True, file_size=None)
            ]
        elif path == 'dir1':
            return [
                MagicMock(path='dir1/a.txt', is_dir=False, file_size=100),
                MagicMock(path='dir1/b.txt', is_dir=False, file_size=200)
            ]
        elif path == 'dir2':
            return [
                MagicMock(path='dir2/c.txt', is_dir=False, file_size=300),
                MagicMock(path='dir2/dir3', is_dir=True, file_size=None)
            ]
        elif path == 'dir2/dir3':
            return [MagicMock(path='dir2/dir3/d.txt', is_dir=False, file_size=400)]
        else:
            return []

    dbx_sdk = pytest.importorskip('databricks.sdk')
    monkeypatch.setattr(dbx_sdk, 'WorkspaceClient', MagicMock())

    mlflow = pytest.importorskip('mlflow')
    mock_mlflow_client = MagicMock()
    monkeypatch.setattr(mlflow, 'MlflowClient', mock_mlflow_client)

    mock_mlflow_client.return_value.get_run.return_value = MagicMock(info=MagicMock(experiment_id=EXPERIMENT_ID))
    mock_mlflow_client.return_value.list_artifacts.side_effect = mock_mlflow_client_list_artifacts

    yield MLFlowObjectStore(DEFAULT_PATH)


def test_get_artifact_path(mlflow_object_store):
    # Relative MLFlow artifact path
    assert mlflow_object_store.get_artifact_path(ARTIFACT_PATH) == ARTIFACT_PATH

    # Absolute DBFS path
    assert mlflow_object_store.get_artifact_path(DEFAULT_PATH + ARTIFACT_PATH) == ARTIFACT_PATH

    # Absolute DBFS path with placeholders
    path = TEST_PATH_FORMAT.format(experiment_id=PLACEHOLDER_EXPERIMENT_ID, run_id=PLACEHOLDER_RUN_ID) + ARTIFACT_PATH
    assert mlflow_object_store.get_artifact_path(path) == ARTIFACT_PATH

    # Raises ValueError for different experiment ID
    path = TEST_PATH_FORMAT.format(experiment_id='different-experiment', run_id=PLACEHOLDER_RUN_ID) + ARTIFACT_PATH
    with pytest.raises(ValueError):
        mlflow_object_store.get_artifact_path(path)

    # Raises ValueError for different run ID
    path = TEST_PATH_FORMAT.format(experiment_id=PLACEHOLDER_EXPERIMENT_ID, run_id='different-run') + ARTIFACT_PATH
    with pytest.raises(ValueError):
        mlflow_object_store.get_artifact_path(path)


def test_get_dbfs_path(mlflow_object_store):
    experiment_id = mlflow_object_store.experiment_id
    run_id = mlflow_object_store.run_id

    expected_dbfs_path = f'databricks/mlflow-tracking/{experiment_id}/{run_id}/artifacts/{ARTIFACT_PATH}'
    assert mlflow_object_store.get_dbfs_path(ARTIFACT_PATH) == expected_dbfs_path


def test_get_uri(mlflow_object_store):
    experiment_id = mlflow_object_store.experiment_id
    run_id = mlflow_object_store.run_id
    expected_uri = f'dbfs:/databricks/mlflow-tracking/{experiment_id}/{run_id}/artifacts/{ARTIFACT_PATH}'

    # Relative MLFlow artifact path
    assert mlflow_object_store.get_uri(ARTIFACT_PATH) == expected_uri

    # Absolute DBFS path
    assert mlflow_object_store.get_uri(DEFAULT_PATH + ARTIFACT_PATH) == expected_uri


def test_get_artifact_info(mlflow_object_store):
    assert mlflow_object_store._get_artifact_info('dir1/a.txt').path == 'dir1/a.txt'
    assert mlflow_object_store._get_artifact_info('dir1/b.txt').path == 'dir1/b.txt'
    assert mlflow_object_store._get_artifact_info('dir2/c.txt').path == 'dir2/c.txt'
    assert mlflow_object_store._get_artifact_info('dir2/dir3/d.txt').path == 'dir2/dir3/d.txt'

    # Test with absolute DBFS path
    assert mlflow_object_store._get_artifact_info(DEFAULT_PATH + 'dir1/a.txt').path == 'dir1/a.txt'

    # Verify directories are not returned
    assert mlflow_object_store._get_artifact_info('dir1') is None

    # Test non-existent artifact
    assert mlflow_object_store._get_artifact_info('nonexistent.txt') is None


def test_get_object_size(mlflow_object_store):
    assert mlflow_object_store.get_object_size('dir1/a.txt') == 100
    assert mlflow_object_store.get_object_size('dir1/b.txt') == 200
    assert mlflow_object_store.get_object_size('dir2/c.txt') == 300
    assert mlflow_object_store.get_object_size('dir2/dir3/d.txt') == 400

    # Test with absolute DBFS path
    assert mlflow_object_store.get_object_size(DEFAULT_PATH + 'dir1/a.txt') == 100

    # Verify FileNotFoundError is raised for non-existent artifact
    with pytest.raises(FileNotFoundError):
        mlflow_object_store.get_object_size('dir1')
    with pytest.raises(FileNotFoundError):
        mlflow_object_store.get_object_size('nonexistent.txt')


def test_download_object(mlflow_object_store, tmp_path):

    def mock_mlflow_client_download_artifacts(*args, **kwargs):
        path = kwargs['path']
        dst_path = kwargs['dst_path']
        local_path = os.path.join(dst_path, path)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)

        size = mlflow_object_store.get_object_size(path)
        file_content = bytes('0' * (size), 'utf-8')

        print(local_path)

        with open(local_path, 'wb') as fp:
            fp.write(file_content)
        return local_path

    mlflow_object_store._mlflow_client.download_artifacts.side_effect = mock_mlflow_client_download_artifacts

    # Test downloading file
    object_name = 'dir1/a.txt'
    file_to_download = str(tmp_path / Path(object_name))
    mlflow_object_store.download_object(object_name, file_to_download)
    assert os.path.exists(file_to_download)
    assert os.path.getsize(file_to_download) == mlflow_object_store.get_object_size(object_name)

    # Test cannot overwrite existing file when `overwrite` is False
    with pytest.raises(FileExistsError):
        mlflow_object_store.download_object(object_name, file_to_download, overwrite=False)

    # Test can overwrite existing file when `overwrite` is True
    mlflow_object_store.download_object(object_name, file_to_download, overwrite=True)

    # Test downloading file under different name
    object_name = 'dir1/a.txt'
    file_to_download = str(tmp_path / Path('renamed.txt'))
    mlflow_object_store.download_object(object_name, file_to_download)
    assert os.path.exists(file_to_download)
    assert os.path.getsize(file_to_download) == mlflow_object_store.get_object_size(object_name)

    # Raises FileNotFound when artifact does not exist
    with pytest.raises(FileNotFoundError):
        mlflow_object_store.download_object('nonexistent.txt', file_to_download)


def test_upload_object(mlflow_object_store, tmp_path):
    file_to_upload = str(tmp_path / Path('file.txt'))
    with open(file_to_upload, 'wb') as f:
        f.write(bytes(range(20)))

    object_name = 'dir1/file.txt'
    mlflow_object_store.upload_object(object_name=object_name, filename=file_to_upload)
    run_id, local_path, artifact_dir = mlflow_object_store._mlflow_client.log_artifact.call_args.args
    assert run_id == mlflow_object_store.run_id
    assert os.path.basename(local_path) == os.path.basename(object_name)
    assert artifact_dir == os.path.dirname(object_name)

    # Test basename symlink is created with correct name when object base name is different
    object_name = 'dir1/renamed.txt'
    mlflow_object_store.upload_object(object_name=object_name, filename=file_to_upload)
    _, local_path, _ = mlflow_object_store._mlflow_client.log_artifact.call_args.args
    assert os.path.basename(local_path) == os.path.basename(object_name)


def test_list_objects(mlflow_object_store):
    expected = {'dir1/a.txt', 'dir1/b.txt', 'dir2/c.txt', 'dir2/dir3/d.txt'}
    assert set(mlflow_object_store.list_objects()) == expected
