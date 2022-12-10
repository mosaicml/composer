# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from composer.utils import OCIObjectStore


@pytest.fixture
def mock_bucket_name():
    return 'my_bucket'


@pytest.fixture
def test_oci_obj_store(mock_bucket_name, monkeypatch):
    oci = pytest.importorskip('oci')

    # Mock all the oci functions.
    mock_config = MagicMock()
    mock_from_file = MagicMock(return_value=mock_config)
    monkeypatch.setattr(oci.config, 'from_file', mock_from_file)
    mock_object_storage_client = MagicMock()
    monkeypatch.setattr(oci.object_storage, 'ObjectStorageClient', mock_object_storage_client)
    mock_upload_manager = MagicMock()
    monkeypatch.setattr(oci.object_storage, 'UploadManager', mock_upload_manager)

    # Create OCIObjectStore object.
    oci_os = OCIObjectStore(mock_bucket_name)
    mock_namespace = 'my_namespace'
    oci_os.namespace = mock_namespace
    return oci_os


def test_upload_object(test_oci_obj_store, monkeypatch, tmp_path, mock_bucket_name):
    pytest.importorskip('oci')
    oci_os = test_oci_obj_store
    mock_object_name = 'my_object'

    mock_upload_file = MagicMock()
    monkeypatch.setattr(oci_os.upload_manager, 'upload_file', mock_upload_file)
    file_to_upload = str(tmp_path / Path('my_upload.bin'))
    with open(file_to_upload, 'wb') as f:
        f.write(bytes(range(20)))

    oci_os.upload_object(object_name=mock_object_name, filename=file_to_upload)
    mock_upload_file.assert_called_once_with(namespace_name=oci_os.namespace,
                                             bucket_name=mock_bucket_name,
                                             object_name=mock_object_name,
                                             file_path=file_to_upload)


def test_download_object(test_oci_obj_store, monkeypatch, tmp_path, mock_bucket_name):
    pytest.importorskip('oci')
    oci_os = test_oci_obj_store
    mock_object_name = 'my_object'
    mock_response_object = MagicMock()
    file_content = bytes(range(4))
    mock_response_object.data.content = file_content
    mock_get_object = MagicMock(return_value=mock_response_object)
    monkeypatch.setattr(oci_os.client, 'get_object', mock_get_object)
    file_to_download_to = str(tmp_path / Path('my_download.bin'))

    oci_os.download_object(object_name=mock_object_name, filename=file_to_download_to)
    mock_get_object.assert_called_once_with(namespace_name=oci_os.namespace,
                                            bucket_name=mock_bucket_name,
                                            object_name=mock_object_name)

    with open(file_to_download_to, 'rb') as f:
        actual_content = f.readline()
    assert actual_content == file_content


def test_get_object_size(test_oci_obj_store, monkeypatch):
    pytest.importorskip('oci')
    oci_os = test_oci_obj_store
    mock_object_name = 'my_object'
    mock_object_size = 11

    mock_response = MagicMock()
    mock_response.status = 200
    mock_response.data.headers = {'Content-Length': mock_object_size}
    mock_get_object_fn = MagicMock(return_value=mock_response)

    monkeypatch.setattr(oci_os.client, 'get_object', mock_get_object_fn)
    assert oci_os.get_object_size(mock_object_name) == mock_object_size

    mock_response.status = 400
    with pytest.raises(FileNotFoundError):
        oci_os.get_object_size(mock_object_name)
