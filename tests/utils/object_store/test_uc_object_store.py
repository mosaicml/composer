# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from unittest import mock
from unittest.mock import ANY, MagicMock

import pytest
from torch.utils.data import DataLoader

from composer.loggers import RemoteUploaderDownloader
from composer.trainer import Trainer
from composer.utils import UCObjectStore
from composer.utils.object_store.object_store import ObjectStoreTransientError
from tests.common import RandomClassificationDataset, SimpleModel


@pytest.fixture
def ws_client(monkeypatch):
    mock_files = MagicMock()
    mock_api_client = MagicMock()
    mock_ws_client = MagicMock()
    monkeypatch.setattr(mock_ws_client, 'files', mock_files)
    monkeypatch.setattr(mock_ws_client, 'api_client', mock_api_client)
    return mock_ws_client


@pytest.fixture
def uc_object_store(ws_client, monkeypatch):
    db = pytest.importorskip('databricks.sdk', reason='requires databricks')

    monkeypatch.setenv('DATABRICKS_HOST', 'test-host')
    monkeypatch.setenv('DATABRICKS_TOKEN', 'test-token')
    with mock.patch.object(db, 'WorkspaceClient', lambda: ws_client):
        yield UCObjectStore(path='Volumes/catalog/schema/volume/path/')


@pytest.mark.skip  # TODO: setup databricks auth on github actions
@pytest.mark.remote
def test_uc_object_store_integration():
    model = SimpleModel()
    train_dataset = RandomClassificationDataset()
    train_dataloader = DataLoader(dataset=train_dataset)
    trainer_save = Trainer(model=model,
                           train_dataloader=train_dataloader,
                           save_folder='dbfs:/Volumes/ml/mosaicml/test-volume/checkpoints/{run_name}',
                           save_filename='test-model.pt',
                           max_duration='1ba')
    run_name = trainer_save.state.run_name
    trainer_save.fit()
    trainer_save.close()

    trainer_load = Trainer(model=model,
                           train_dataloader=train_dataloader,
                           load_path=f'dbfs:/Volumes/ml/mosaicml/test-volume/checkpoints/{run_name}/test-model.pt',
                           max_duration='2ba')
    trainer_load.fit()
    trainer_load.close()


def test_uc_object_store_without_env():
    with pytest.raises(ValueError):
        UCObjectStore(path='Volumes/test-volume/')


def test_uc_object_store_invalid_prefix(monkeypatch):
    monkeypatch.setenv('DATABRICKS_HOST', 'test-host')
    monkeypatch.setenv('DATABRICKS_TOKEN', 'test-token')

    with pytest.raises(ValueError):
        UCObjectStore(path='root/')
    with pytest.raises(ValueError):
        UCObjectStore(path='uc://Volumes')
    with pytest.raises(ValueError):
        UCObjectStore(path='Volumes/catalog/schema/')


@pytest.mark.parametrize('result', ['success', 'not_found'])
def test_get_object_size(ws_client, uc_object_store, result: str):
    if result == 'success':
        db_files = pytest.importorskip('databricks.sdk.service.files')
        ws_client.files.get_status.return_value = db_files.FileInfo(file_size=100)
        assert uc_object_store.get_object_size('train.txt') == 100
    elif result == 'not_found':
        db_core = pytest.importorskip('databricks.sdk.core', reason='requires databricks')
        ws_client.files.get_status.side_effect = db_core.DatabricksError('The file being accessed is not found',
                                                                         error_code='NOT_FOUND')
        with pytest.raises(FileNotFoundError):
            uc_object_store.get_object_size('train.txt')
    else:
        raise NotImplementedError(f'Test for result={result} is not implemented.')


def test_get_uri(uc_object_store):
    assert uc_object_store.get_uri('train.txt') == 'dbfs:/Volumes/catalog/schema/volume/train.txt'
    assert uc_object_store.get_uri('Volumes/catalog/schema/volume/checkpoint/model.bin'
                                  ) == 'dbfs:/Volumes/catalog/schema/volume/checkpoint/model.bin'


def test_upload_object(ws_client, uc_object_store, tmp_path):
    file_to_upload = str(tmp_path / Path('train.txt'))
    with open(file_to_upload, 'wb') as f:
        f.write(bytes(range(20)))

    uc_object_store.upload_object(object_name='path/train.txt', filename=file_to_upload)
    ws_client.files.upload.assert_called_with('/Volumes/catalog/schema/volume/path/train.txt', ANY)


@pytest.mark.parametrize('result', ['success', 'file_exists', 'overwrite_file', 'not_found', 'error'])
def test_download_object(ws_client, uc_object_store, tmp_path, result: str):

    object_name = 'remote-model.bin'
    file_content = bytes('0' * (100), 'utf-8')
    file_to_download = str(tmp_path / Path('model.bin'))

    def generate_dummy_file(_):
        db_files = pytest.importorskip('databricks.sdk.service.files')
        with open(file_to_download, 'wb') as fp:
            fp.write(file_content)
        f = open(file_to_download, 'rb')
        return db_files.DownloadResponse(contents=f)

    if result == 'success':
        ws_client.files.download.side_effect = generate_dummy_file
        uc_object_store.download_object(object_name, filename=file_to_download)
        ws_client.files.download.assert_called_with('/Volumes/catalog/schema/volume/remote-model.bin')

    elif result == 'file_exists':
        with open(file_to_download, 'wb') as fp:
            fp.write(bytes('1' * (100), 'utf-8'))
        with pytest.raises(FileExistsError):
            uc_object_store.download_object(object_name, file_to_download)

    elif result == 'overwrite_file':
        with open(file_to_download, 'wb') as fp:
            fp.write(bytes('1' * (100), 'utf-8'))
        ws_client.files.download.side_effect = generate_dummy_file
        uc_object_store.download_object(object_name, file_to_download, overwrite=True)
        ws_client.files.download.assert_called_with('/Volumes/catalog/schema/volume/remote-model.bin')

        # verify that the file was actually overwritten
        with open(file_to_download, 'rb') as f:
            actual_content = f.readline()
        assert actual_content == file_content

    elif result == 'not_found':
        db_core = pytest.importorskip('databricks.sdk.core', reason='requires databricks')
        ws_client.files.download.side_effect = db_core.DatabricksError('The file being accessed is not found',
                                                                       error_code='NOT_FOUND')
        with pytest.raises(FileNotFoundError):
            uc_object_store.download_object(object_name, file_to_download)

    elif result == 'error':
        db_core = pytest.importorskip('databricks.sdk.core', reason='requires databricks')
        ws_client.files.download.side_effect = db_core.DatabricksError

        with pytest.raises(ObjectStoreTransientError):
            uc_object_store.download_object(object_name, file_to_download)
    else:
        raise NotImplementedError(f'Test for result={result} is not implemented.')


@pytest.mark.parametrize('result', ['success', 'prefix_none', 'not_found', 'error'])
def test_list_objects(ws_client, uc_object_store, result):
    expected_files = [
        '/Volumes/catalog/volume/schema/path/to/folder/file1.txt',
        '/Volumes/catalog/volume/schema/path/to/folder/file2.txt',
    ]
    uc_list_api_response = {
        'files': [{
            'path': '/Volumes/catalog/volume/schema/path/to/folder/file1.txt',
            'is_dir': False
        }, {
            'path': '/Volumes/catalog/volume/schema/path/to/folder/file2.txt',
            'is_dir': False
        }, {
            'path': '/Volumes/catalog/volume/schema/path/to/folder/samples/',
            'is_dir': True
        }]
    }

    prefix = 'Volumes/catalog/schema/volume/path/to/folder'

    if result == 'success':
        ws_client.api_client.do.return_value = uc_list_api_response
        actual_files = uc_object_store.list_objects(prefix=prefix)

        assert actual_files == expected_files
        ws_client.api_client.do.assert_called_once_with(
            method='GET',
            path=uc_object_store._UC_VOLUME_LIST_API_ENDPOINT,
            data='{"path": "/Volumes/catalog/schema/volume/path/to/folder"}',
            headers={'Source': 'mosaicml/composer'})

    elif result == 'prefix_none':
        ws_client.api_client.do.return_value = uc_list_api_response
        actual_files = uc_object_store.list_objects(prefix=None)

        assert actual_files == expected_files
        ws_client.api_client.do.assert_called_once_with(method='GET',
                                                        path=uc_object_store._UC_VOLUME_LIST_API_ENDPOINT,
                                                        data='{"path": "/Volumes/catalog/schema/volume/."}',
                                                        headers={'Source': 'mosaicml/composer'})

    elif result == 'not_found':
        db_core = pytest.importorskip('databricks.sdk.core', reason='requires databricks')
        ws_client.api_client.do.side_effect = db_core.DatabricksError(
            'The path you provided does not exist or is not a directory.', error_code='NOT_FOUND')
        with pytest.raises(FileNotFoundError):
            uc_object_store.list_objects(prefix=prefix)

    elif result == 'error':
        db_core = pytest.importorskip('databricks.sdk.core', reason='requires databricks')
        ws_client.api_client.do.side_effect = db_core.DatabricksError

        with pytest.raises(ObjectStoreTransientError):
            uc_object_store.list_objects(prefix=prefix)

    else:
        raise NotImplementedError(f'Test for result={result} is not implemented.')


def test_uc_object_store_with_remote_ud(uc_object_store):
    uri = 'dbfs:/Volumes/path/to/my/folder/'
    rud = RemoteUploaderDownloader(bucket_uri=uri, backend_kwargs={'path': 'Volumes/catalog/schema/volume/path'})
    assert isinstance(rud.remote_backend, UCObjectStore)
