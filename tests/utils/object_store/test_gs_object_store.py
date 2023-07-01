# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import os
from unittest import mock

import pytest
from botocore.exceptions import ClientError
from google.cloud import storage

from composer.utils import GsObjectStore


@pytest.fixture
def gs_object_store(monkeypatch):
    with mock.patch.object(storage, 'Client', return_value=mock.MagicMock()):
        yield GsObjectStore(bucket='test-bucket', prefix='test-prefix')


def test_get_uri(gs_object_store):
    object_name = 'test-object'
    expected_uri = 'gs://test-bucket/test-prefix/test-object'
    assert (gs_object_store.get_uri(object_name) == expected_uri)


def test_get_key(gs_object_store):
    object_name = 'test-object'
    expected_key = 'test-prefix/test-object'
    assert (gs_object_store.get_key(object_name) == expected_key)


def test_get_object_size(gs_object_store, monkeypatch):
    mock_bucket = mock.MagicMock()
    mock_blob = mock.MagicMock()
    mock_blob.size = 100

    monkeypatch.setattr(gs_object_store.client, 'get_bucket', mock.MagicMock(return_value=mock_bucket))

    mock_bucket.get_blob.return_value = mock_blob

    object_name = 'test-object'
    assert (gs_object_store.get_object_size(object_name) == 100)


def test_upload_object(gs_object_store, monkeypatch):
    mock_bucket = mock.MagicMock()
    mock_blob = mock.MagicMock()

    monkeypatch.setattr(gs_object_store.client, 'bucket', mock.MagicMock(return_value=mock_bucket))
    mock_bucket.blob.return_value = mock_blob

    source_file_name = 'dummy-file.txt'
    destination_blob_name = 'dummy-blob.txt'

    gs_object_store.upload_blob('test-bucket', source_file_name, destination_blob_name)

    mock_blob.upload_from_filename.assert_called_with(source_file_name, if_generation_match=0)
    assert mock_blob.upload_from_filename.call_count == 1


@pytest.mark.parametrize('result', ['success', 'file_exists', 'error'])
def test_download_object(gs_object_store, monkeypatch, tmp_path, result: str):
    mock_bucket = mock.MagicMock()
    mock_blob = mock.MagicMock()
    mock_os = mock.MagicMock()

    object_name = 'test-object'
    filename = 'test-file.txt'

    def generate_dummy_file(x):
        with open(x, 'wb') as fp:
            fp.write(bytes('0' * (1024 * 1024 * 1024), 'utf-8'))

    monkeypatch.setattr(gs_object_store.client, 'bucket', mock.MagicMock(return_value=mock_bucket))
    mock_bucket.blob.return_value = mock_blob
    mock_blob.download_to_filename.side_effect = generate_dummy_file

    if result == 'success':
        gs_object_store.download_blob(object_name, filename, overwrite=True)
        mock_blob.download_to_filename.assert_called_once_with(mock.ANY)

    elif result == 'file_exists':
        monkeypatch.setattr(os.path, 'exists', mock.MagicMock(return_value=True))
        with pytest.raises(FileExistsError):
            gs_object_store.download_blob(object_name, filename)
        assert not mock_os.rename.called

    else:  # error
        mock_blob.download_to_filename.side_effect = ClientError({}, 'operation')
        with pytest.raises(ClientError):
            gs_object_store.download_blob(object_name, filename, overwrite=True)
