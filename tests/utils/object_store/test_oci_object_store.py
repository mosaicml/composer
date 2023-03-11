# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from unittest.mock import MagicMock, Mock

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


@pytest.mark.parametrize('result', ['success', 'bucket_not_found'])
def test_upload_object(test_oci_obj_store, monkeypatch, tmp_path, mock_bucket_name, result: str):
    oci = pytest.importorskip('oci')
    oci_os = test_oci_obj_store
    mock_object_name = 'my_object'

    file_to_upload = str(tmp_path / Path('my_upload.bin'))
    with open(file_to_upload, 'wb') as f:
        f.write(bytes(range(20)))

    if result == 'success':
        with monkeypatch.context() as m:
            mock_upload_file = MagicMock()
            m.setattr(oci_os.upload_manager, 'upload_file', mock_upload_file)
            oci_os.upload_object(object_name=mock_object_name, filename=file_to_upload)
            mock_upload_file.assert_called_once_with(namespace_name=oci_os.namespace,
                                                     bucket_name=mock_bucket_name,
                                                     object_name=mock_object_name,
                                                     file_path=file_to_upload)
    else:  # result = bucket_not_found
        bucket_not_found_msg = f'Either the bucket named f{mock_bucket_name} does not exist in the namespace*'
        mock_upload_file_with_exception = Mock(side_effect=oci.exceptions.ServiceError(
            status=404, code='BucketNotFound', headers={'opc-request-id': 'foo'}, message=bucket_not_found_msg))
        with monkeypatch.context() as m:
            m.setattr(oci_os.upload_manager, 'upload_file', mock_upload_file_with_exception)
            with pytest.raises(
                    ValueError,
                    match=
                    f'Bucket specified in oci://{mock_bucket_name}/{mock_object_name} not found. {bucket_not_found_msg}'
            ):
                oci_os.upload_object(mock_object_name, filename=file_to_upload)


@pytest.mark.parametrize('result', ['success', 'file_exists', 'obj_not_found', 'bucket_not_found'])
def test_download_object(test_oci_obj_store, monkeypatch, tmp_path, mock_bucket_name, result: str):
    oci = pytest.importorskip('oci')
    oci_os = test_oci_obj_store
    mock_object_name = 'my_object'

    if result == 'success':
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

    elif result == 'file_exists':
        file = tmp_path / Path('file_exists.bin')
        file.touch()
        filename = str(file)
        with pytest.raises(FileExistsError,
                           match=f'The file at {filename} already exists and overwrite is set to False'):
            oci_os.download_object(mock_object_name, filename=filename)

    elif result == 'obj_not_found':
        file_to_download_to = str(tmp_path / Path('my_obj_not_found_file.bin'))
        obj_not_found_msg = f"The object '{mock_object_name}' was not found in the bucket f'{mock_bucket_name}'"
        mock_get_object_fn_with_exception = Mock(side_effect=oci.exceptions.ServiceError(
            status=404, code='ObjectNotFound', headers={'opc-request-id': 'foo'}, message=obj_not_found_msg))
        with monkeypatch.context() as m:
            m.setattr(oci_os.client, 'get_object', mock_get_object_fn_with_exception)
            with pytest.raises(
                    FileNotFoundError,
                    match=f'Object oci://{mock_bucket_name}/{mock_object_name} not found. {obj_not_found_msg}'):
                oci_os.download_object(mock_object_name, filename=file_to_download_to)
    else:  #result == 'bucket_not_found':
        file_to_download_to = str(tmp_path / Path('my_bucket_not_found_file.bin'))
        bucket_not_found_msg = f'Either the bucket named f{mock_bucket_name} does not exist in the namespace*'
        mock_get_object_fn_with_exception = Mock(side_effect=oci.exceptions.ServiceError(
            status=404, code='BucketNotFound', headers={'opc-request-id': 'foo'}, message=bucket_not_found_msg))
        with monkeypatch.context() as m:
            m.setattr(oci_os.client, 'get_object', mock_get_object_fn_with_exception)
            with pytest.raises(
                    ValueError,
                    match=
                    f'Bucket specified in oci://{mock_bucket_name}/{mock_object_name} not found. {bucket_not_found_msg}'
            ):
                oci_os.download_object(mock_object_name, filename=file_to_download_to)


@pytest.mark.parametrize('result', ['success', 'obj_not_found', 'bucket_not_found'])
def test_get_object_size(test_oci_obj_store, mock_bucket_name, monkeypatch, result: str):
    oci = pytest.importorskip('oci')
    oci_os = test_oci_obj_store
    mock_object_name = 'my_object'
    mock_object_size = 11

    if result == 'success':
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.data.headers = {'Content-Length': mock_object_size}
        mock_get_object_fn = MagicMock(return_value=mock_response)
        with monkeypatch.context() as m:
            m.setattr(oci_os.client, 'get_object', mock_get_object_fn)
            assert oci_os.get_object_size(mock_object_name) == mock_object_size

    elif result == 'obj_not_found':
        obj_not_found_msg = f"The object '{mock_object_name}' was not found in the bucket f'{mock_bucket_name}'"
        mock_get_object_fn_with_exception = Mock(side_effect=oci.exceptions.ServiceError(
            status=404, code='ObjectNotFound', headers={'opc-request-id': 'foo'}, message=obj_not_found_msg))
        with monkeypatch.context() as m:
            m.setattr(oci_os.client, 'get_object', mock_get_object_fn_with_exception)
            with pytest.raises(
                    FileNotFoundError,
                    match=f'Object oci://{mock_bucket_name}/{mock_object_name} not found. {obj_not_found_msg}'):
                oci_os.get_object_size(mock_object_name)

    else:  #result == 'bucket_not_found':
        bucket_not_found_msg = f'Either the bucket named f{mock_bucket_name} does not exist in the namespace*'
        mock_get_object_fn_with_exception = Mock(side_effect=oci.exceptions.ServiceError(
            status=404, code='BucketNotFound', headers={'opc-request-id': 'foo'}, message=bucket_not_found_msg))
        with monkeypatch.context() as m:
            m.setattr(oci_os.client, 'get_object', mock_get_object_fn_with_exception)
            with pytest.raises(
                    ValueError,
                    match=
                    f'Bucket specified in oci://{mock_bucket_name}/{mock_object_name} not found. {bucket_not_found_msg}'
            ):
                oci_os.get_object_size(mock_object_name)
