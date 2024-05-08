# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import os
from unittest import mock
from unittest.mock import MagicMock

import pytest
from botocore.exceptions import ClientError
from torch.utils.data import DataLoader

from composer.loggers import RemoteUploaderDownloader
from composer.optim import DecoupledSGDW
from composer.trainer import Trainer
from composer.utils import GCSObjectStore
from tests.common import RandomClassificationDataset, SimpleModel


def get_gcs_os_from_trainer(trainer: Trainer) -> GCSObjectStore:
    rud = [dest for dest in trainer.logger.destinations if isinstance(dest, RemoteUploaderDownloader)][0]
    gcs_os = rud.remote_backend
    assert isinstance(gcs_os, GCSObjectStore)
    return gcs_os


@pytest.mark.gpu  # json auth is hard to set up on github actions / CPU tests
@pytest.mark.remote
def test_gs_object_store_integration_hmac_auth(expected_use_gcs_sdk_val=False, client_should_be_none=True):
    model = SimpleModel()
    train_dataset = RandomClassificationDataset()
    train_dataloader = DataLoader(dataset=train_dataset)
    optimizer = DecoupledSGDW(model.parameters(), lr=1e-4)
    trainer_save = Trainer(
        model=model,
        optimizers=optimizer,
        train_dataloader=train_dataloader,
        save_folder='gs://mosaicml-internal-integration-testing/checkpoints/{run_name}',
        save_filename='test-model.pt',
        max_duration='1ba',
        precision='amp_bf16',
    )
    run_name = trainer_save.state.run_name
    gcs_os = get_gcs_os_from_trainer(trainer_save)
    assert gcs_os.use_gcs_sdk == expected_use_gcs_sdk_val
    if client_should_be_none:
        assert gcs_os.client is None
    else:
        assert gcs_os.client is not None
    trainer_save.fit()
    trainer_save.close()

    trainer_load = Trainer(
        model=model,
        optimizers=optimizer,
        train_dataloader=train_dataloader,
        load_path=f'gs://mosaicml-internal-integration-testing/checkpoints/{run_name}/test-model.pt',
        max_duration='2ba',
        precision='amp_bf16',
    )
    trainer_load.fit()
    trainer_load.close()


@pytest.mark.gpu
@pytest.mark.remote
def test_gs_object_store_integration_json_auth():
    with mock.patch.dict(os.environ):
        if 'GCS_KEY' in os.environ:
            del os.environ['GCS_KEY']
        if 'GCS_SECRET' in os.environ:
            del os.environ['GCS_SECRET']
        test_gs_object_store_integration_json_auth(expected_use_gcs_sdk_val=True, client_should_be_none=False)


@pytest.fixture
def gs_object_store(monkeypatch):
    from google import auth  # type: ignore
    from google.cloud.storage import Client
    monkeypatch.delenv('GCS_KEY', raising=False)
    monkeypatch.delenv('GCS_SECRET', raising=False)
    with mock.patch.object(auth, 'default', return_value=(None, None)):
        with mock.patch.object(Client, '__init__', return_value=None):
            with mock.patch.object(Client, 'get_bucket', return_value=mock.MagicMock()):
                gcs_object_store = GCSObjectStore(bucket='test-bucket', prefix='test-prefix')
                gcs_object_store.client = mock.MagicMock()
                yield gcs_object_store


def test_get_uri(gs_object_store):
    object_name = 'test-object'
    expected_uri = 'gs://test-bucket/test-prefix/test-object'
    assert (gs_object_store.get_uri(object_name) == expected_uri)


def test_get_key(gs_object_store):
    object_name = 'test-object'
    expected_key = 'test-prefix/test-object'
    assert (gs_object_store.get_key(object_name) == expected_key)


def test_get_object_size(gs_object_store, monkeypatch):
    mock_blob = mock.MagicMock()
    mock_blob.size = 100

    monkeypatch.setattr(gs_object_store.bucket, 'get_blob', mock.MagicMock(return_value=mock_blob))

    object_name = 'test-object'
    assert (gs_object_store.get_object_size(object_name) == 100)


def test_upload_object(gs_object_store, monkeypatch):
    mock_blob = mock.MagicMock()

    monkeypatch.setattr(gs_object_store.bucket, 'blob', mock.MagicMock(return_value=mock_blob))

    source_file_name = 'dummy-file.txt'
    destination_blob_name = 'dummy-blob.txt'

    gs_object_store.upload_object(destination_blob_name, source_file_name)

    from google.cloud.storage.retry import DEFAULT_RETRY
    mock_blob.upload_from_filename.assert_called_with(source_file_name, retry=DEFAULT_RETRY)
    assert mock_blob.upload_from_filename.call_count == 1


@pytest.mark.parametrize('result', ['success', 'file_exists', 'error'])
def test_download_object(gs_object_store, monkeypatch, tmp_path, result: str):
    mock_blob = mock.MagicMock()
    mock_os = mock.MagicMock()

    object_name = 'test-object'
    filename = 'test-file.txt'

    def generate_dummy_file(x):
        with open(x, 'wb') as fp:
            fp.write(bytes('0' * (10), 'utf-8'))

    monkeypatch.setattr(gs_object_store.bucket, 'blob', mock.MagicMock(return_value=mock_blob))
    mock_blob.download_to_filename.side_effect = generate_dummy_file

    if result == 'success':
        gs_object_store.download_object(object_name, filename, overwrite=True)
        mock_blob.download_to_filename.assert_called_once_with(mock.ANY)

    elif result == 'file_exists':
        monkeypatch.setattr(os.path, 'exists', mock.MagicMock(return_value=True))
        with pytest.raises(FileExistsError):
            gs_object_store.download_object(object_name, filename)
        assert not mock_os.rename.called

    else:  # error
        mock_blob.download_to_filename.side_effect = ClientError({}, 'operation')
        with pytest.raises(ClientError):
            gs_object_store.download_object(object_name, filename, overwrite=True)


@pytest.mark.parametrize('result', ['success', 'prefix_not_found'])
def test_list_objects_success(gs_object_store, monkeypatch, result: str):
    if result == 'success':
        blob1_mock = MagicMock(spec=['name'])
        blob1_mock.name = 'path/to/object1'
        blob2_mock = MagicMock(spec=['name'])
        blob2_mock.name = 'path/to/object2'

        bucket_mock = MagicMock()
        bucket_mock.list_blobs.return_value = [blob1_mock, blob2_mock]

        gs_object_store.bucket = bucket_mock

        actual = gs_object_store.list_objects(prefix='path/to')

        bucket_mock.list_blobs.assert_called_once_with(prefix='test-prefix/path/to')
        assert (actual == ['path/to/object1', 'path/to/object2'])

    elif result == 'prefix_not_found':

        bucket_mock = MagicMock()
        bucket_mock.list_blobs.return_value = []

        gs_object_store.bucket = bucket_mock
        actual = gs_object_store.list_objects(prefix='non_existent_prefix')

        assert (actual == [])
        bucket_mock.list_blobs.assert_called_once_with(prefix='test-prefix/non_existent_prefix')
