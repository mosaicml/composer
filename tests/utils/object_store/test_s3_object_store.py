# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import pathlib

import moto
import pytest

from composer.utils.object_store.s3_object_store import S3ObjectStore
from tests.utils.object_store.object_store_settings import object_store_kwargs

try:
    import boto3
    _BOTO3_INSTALLED = True
    del boto3
except ImportError:
    _BOTO3_INSTALLED = False


class MockCallback:

    def __init__(self, total_num_bytes: int) -> None:
        self.total_num_bytes = total_num_bytes
        self.transferred_bytes = 0
        self.num_calls = 0

    def __call__(self, transferred: int, total: int):
        self.num_calls += 1
        assert transferred >= self.transferred_bytes, "transferred should be monotonically increasing"
        self.transferred_bytes = transferred
        assert total == self.total_num_bytes

    def assert_all_data_transferred(self):
        assert self.total_num_bytes == self.transferred_bytes


@pytest.mark.skipif(not _BOTO3_INSTALLED, reason="boto3 is not available")
class TestS3ObjectStore:

    @pytest.fixture
    def s3(self, monkeypatch: pytest.MonkeyPatch):
        import boto3
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", 'testing')
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", 'testing')
        monkeypatch.setenv("AWS_SECURITY_TOKEN", 'testing')
        monkeypatch.setenv("AWS_SESSION_TOKEN", 'testing')
        monkeypatch.setenv("AWS_DEFAULT_REGION", 'us-east-1')
        with moto.mock_s3():
            # create the dummy bucket
            s3 = boto3.client('s3')
            s3.create_bucket(Bucket=object_store_kwargs[S3ObjectStore]['bucket'])

            # Yield our object store
            yield S3ObjectStore(**object_store_kwargs[S3ObjectStore])

    @pytest.fixture
    def dummy_obj(self, s3: S3ObjectStore, tmp_path: pathlib.Path):
        tmpfile_path = tmp_path / "file_to_upload"
        with open(tmpfile_path, "w+") as f:
            f.write("dummy content")
        return tmpfile_path

    def test_upload(self, s3: S3ObjectStore, dummy_obj: pathlib.Path):
        object_name = "tmpfile_object_name"
        cb = MockCallback(dummy_obj.stat().st_size)
        s3.upload_object(object_name, str(dummy_obj), callback=cb)
        cb.assert_all_data_transferred()

    def test_get_uri(self, s3: S3ObjectStore):
        assert s3.get_uri("tmpfile_object_name") == "s3://my-bucket/tmpfile_object_name"

    def test_get_file_size(self, s3: S3ObjectStore, dummy_obj: pathlib.Path):
        object_name = "tmpfile_object_name"
        s3.upload_object(object_name, str(dummy_obj))
        assert s3.get_object_size(object_name) == dummy_obj.stat().st_size

    def test_get_file_size_not_found(self, s3: S3ObjectStore):
        with pytest.raises(FileNotFoundError):
            s3.get_object_size("not found object")

    def test_download(self, s3: S3ObjectStore, dummy_obj: pathlib.Path, tmp_path: pathlib.Path):
        object_name = "tmpfile_object_name"
        s3.upload_object(object_name, str(dummy_obj))
        filepath = str(tmp_path / "destination_path")
        cb = MockCallback(dummy_obj.stat().st_size)
        s3.download_object(object_name, filepath, callback=cb)
        cb.assert_all_data_transferred()

    def test_download_not_found(self, s3: S3ObjectStore):
        with pytest.raises(FileNotFoundError):
            s3.download_object("not_found_object", filepath="not used")
