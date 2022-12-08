# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from composer.utils import OCIObjectStore


@pytest.fixture
def setup_oci_mocks(monkeypatch):
    oci = pytest.importorskip('oci')
    mock_config = MagicMock()
    mock_from_file = MagicMock(return_value=mock_config)
    monkeypatch.setattr(oci.config, 'from_file', mock_from_file)
    mock_object_storage_client = MagicMock()
    monkeypatch.setattr(oci.object_storage, 'ObjectStorageClient', mock_object_storage_client)
    mock_upload_manager = MagicMock()
    monkeypatch.setattr(oci.object_storage, 'UploadManager', mock_upload_manager)


class TestOCIObjectStore:

    @classmethod
    def setup_class(cls):
        pytest.importorskip('oci')
        cls.mock_bucket_name = 'my_bucket'
        cls.mock_namespace = 'my_namespace'

        cls.oci_os = OCIObjectStore(cls.mock_bucket_name)
        cls.oci_os.namespace = cls.mock_namespace

    def test_upload_object(self, setup_oci_mocks, monkeypatch, tmp_path):
        pytest.importorskip('oci')
        mock_object_name = 'my_object'

        # oci_os = OCIObjectStore(mock_bucket_name)
        self.oci_os.namespace = self.mock_namespace
        mock_upload_file = MagicMock()
        monkeypatch.setattr(self.oci_os.upload_manager, 'upload_file', mock_upload_file)
        file_to_upload = str(tmp_path / Path('my_upload.bin'))
        with open(file_to_upload, 'wb') as f:
            f.write(bytes(range(20)))

        self.oci_os.upload_object(object_name=mock_object_name, filename=file_to_upload)
        mock_upload_file.assert_called_once_with(namespace_name=self.mock_namespace,
                                                 bucket_name=self.mock_bucket_name,
                                                 object_name=mock_object_name,
                                                 file_path=file_to_upload)

    def test_download_object(self, monkeypatch, tmp_path):
        pytest.importorskip('oci')
        mock_object_name = 'my_object'
        mock_response_object = MagicMock()
        file_content = bytes(range(4))
        mock_response_object.data.content = file_content
        mock_get_object = MagicMock(return_value=mock_response_object)
        monkeypatch.setattr(self.oci_os.client, 'get_object', mock_get_object)
        file_to_download_to = str(tmp_path / Path('my_download.bin'))

        self.oci_os.download_object(object_name=mock_object_name, filename=file_to_download_to)
        mock_get_object.assert_called_once_with(namespace_name=self.mock_namespace,
                                                bucket_name=self.mock_bucket_name,
                                                object_name=mock_object_name)

        with open(file_to_download_to, 'rb') as f:
            actual_content = f.readline()
        assert actual_content == file_content

    def test_get_object_size(self, monkeypatch):
        pytest.importorskip('oci')
        mock_object_name = 'my_object'
        mock_object_size = 11
        mock_object_1, mock_object_2 = MagicMock(), MagicMock()
        mock_object_1.name = mock_object_name
        mock_object_2.name = 'foobar'
        mock_object_1.size = mock_object_size
        mock_object_2.size = 3

        mock_list_objects_return = MagicMock()
        mock_list_objects_return.data.objects = [mock_object_1, mock_object_2]
        mock_list_objects_fn = MagicMock(return_value=mock_list_objects_return)
        monkeypatch.setattr(self.oci_os.client, 'list_objects', mock_list_objects_fn)

        assert self.oci_os.get_object_size(mock_object_name) == mock_object_size


# def test_checkpointing_with_oci_object_store(self, setup_oci_mocks, monkeypatch, tmp_path):
#     pass
