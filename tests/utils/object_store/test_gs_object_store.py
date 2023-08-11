# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import os
from unittest import mock
from unittest.mock import MagicMock

import pytest
from botocore.exceptions import ClientError

from composer.utils import GCSObjectStore


@pytest.fixture
def gs_object_store(monkeypatch):
    from google.cloud.storage import Client
    with mock.patch.dict(os.environ, {'GOOGLE_APPLICATION_CREDENTIALS': 'FAKE_CREDENTIAL'}):
        mock_client = mock.MagicMock()
        with mock.patch.object(Client, 'from_service_account_json', return_value=mock_client):
            yield GCSObjectStore('gs://test-bucket/test-prefix/')


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

    gs_object_store.upload_object(source_file_name, destination_blob_name)

    mock_blob.upload_from_filename.assert_called_with(source_file_name)
    assert mock_blob.upload_from_filename.call_count == 1


@pytest.mark.parametrize('result', ['success', 'file_exists', 'error'])
def test_download_object(gs_object_store, monkeypatch, tmp_path, result: str):
    mock_blob = mock.MagicMock()
    mock_os = mock.MagicMock()

    object_name = 'test-object'
    filename = 'test-file.txt'

    def generate_dummy_file(x):
        with open(x, 'wb') as fp:
            fp.write(bytes('0' * (1024 * 1024 * 1024), 'utf-8'))

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
