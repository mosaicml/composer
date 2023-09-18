from pathlib import Path
from unittest import mock
from unittest.mock import ANY, MagicMock

import pytest

from composer.utils import UCObjectStore


@pytest.fixture
def ws_client(monkeypatch):
    mock_files = MagicMock()
    mock_ws_client = MagicMock()
    monkeypatch.setattr(mock_ws_client, 'files', mock_files)
    return mock_ws_client


@pytest.fixture
def uc_object_store(ws_client, monkeypatch):
    db = pytest.importorskip('databricks.sdk', reason='requires databricks')

    monkeypatch.setenv('DATABRICKS_HOST', 'test-host')
    monkeypatch.setenv('DATABRICKS_TOKEN', 'test-token')
    with mock.patch.object(db, 'WorkspaceClient', lambda: ws_client):
        yield UCObjectStore(uri='uc://Volumes/test-volume/')


def test_uc_object_store_without_env():
    with pytest.raises(ValueError):
        UCObjectStore(uri='uc://Volumes/test-volume/')


def test_get_object_size(ws_client, uc_object_store):
    db_files = pytest.importorskip('databricks.sdk.service.files')
    ws_client.files.get_status.return_value = db_files.FileInfo(file_size=100)
    assert uc_object_store.get_object_size('train.txt') == 100


def test_get_uri(uc_object_store):
    assert uc_object_store.get_uri('train.txt') == 'uc://Volumes/test-volume/train.txt'


def test_upload_object(ws_client, uc_object_store, tmp_path):
    file_to_upload = str(tmp_path / Path('train.txt'))
    with open(file_to_upload, 'wb') as f:
        f.write(bytes(range(20)))

    uc_object_store.upload_object(object_name='train.txt', filename=file_to_upload)
    ws_client.files.upload.assert_called_with('/Volumes/test-volume/train.txt', ANY)


@pytest.mark.parametrize('result', ['success', 'file_exists', 'overwrite_file', 'error'])
def test_download_object(ws_client, uc_object_store, tmp_path, result: str):

    object_name = 'remote-model.bin'
    file_content = bytes('0' * (1024 * 1024 * 1024), 'utf-8')
    file_to_download = str(tmp_path / Path('model.bin'))

    def generate_dummy_file(x):
        db_files = pytest.importorskip('databricks.sdk.service.files')
        with open(file_to_download, 'wb') as fp:
            fp.write(file_content)
        f = open(file_to_download, 'rb')
        return db_files.DownloadResponse(contents=f)

    if result == 'success':
        ws_client.files.download.side_effect = generate_dummy_file
        uc_object_store.download_object(object_name, filename=file_to_download)
        ws_client.files.download.assert_called_with('/Volumes/test-volume/remote-model.bin')

    elif result == 'file_exists':
        with open(file_to_download, 'wb') as fp:
            fp.write(bytes('1' * (1024 * 1024 * 1024), 'utf-8'))
        with pytest.raises(FileExistsError):
            uc_object_store.download_object(object_name, file_to_download)

    elif result == 'overwrite_file':
        with open(file_to_download, 'wb') as fp:
            fp.write(bytes('1' * (1024 * 1024 * 1024), 'utf-8'))
        ws_client.files.download.side_effect = generate_dummy_file
        uc_object_store.download_object(object_name, file_to_download, overwrite=True)
        ws_client.files.download.assert_called_with('/Volumes/test-volume/remote-model.bin')

        # verify that the file was actually overwritten
        with open(file_to_download, 'rb') as f:
            actual_content = f.readline()
        assert actual_content == file_content

    else:  # error
        db_core = pytest.importorskip('databricks.sdk.core', reason='requires databricks')
        ws_client.files.download.side_effect = db_core.DatabricksError

        with pytest.raises(db_core.DatabricksError):
            uc_object_store.download_object(object_name, file_to_download)
