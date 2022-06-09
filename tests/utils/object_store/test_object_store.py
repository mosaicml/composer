# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import os
import pathlib
from typing import Generator

import moto
import pytest

from composer.utils.object_store import LibcloudObjectStore, ObjectStore, S3ObjectStore
from tests.utils.object_store.object_store_settings import object_store_kwargs


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


@pytest.fixture
def object_store(request, monkeypatch: pytest.MonkeyPatch,
                 tmp_path: pathlib.Path) -> Generator[ObjectStore, None, None]:
    if request.param is S3ObjectStore:
        pytest.importorskip("boto3")
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
            yield request.param(**object_store_kwargs[request.param])
    elif request.param is LibcloudObjectStore:
        remote_dir = tmp_path / "remote_dir"
        os.makedirs(remote_dir)
        object_store_kwargs[request.param]['provider_kwargs']['key'] = remote_dir
        yield request.param(**object_store_kwargs[request.param])
    else:
        raise NotImplementedError("Parameterization not implemented")


@pytest.mark.parametrize("object_store", [S3ObjectStore, LibcloudObjectStore], indirect=True)
class TestObjectStore:

    @pytest.fixture
    def dummy_obj(self, object_store: ObjectStore, tmp_path: pathlib.Path):
        tmpfile_path = tmp_path / "file_to_upload"
        with open(tmpfile_path, "w+") as f:
            f.write("dummy content")
        return tmpfile_path

    def test_upload(self, object_store: ObjectStore, dummy_obj: pathlib.Path):
        object_name = "tmpfile_object_name"
        cb = MockCallback(dummy_obj.stat().st_size)
        object_store.upload_object(object_name, str(dummy_obj), callback=cb)
        cb.assert_all_data_transferred()

    def test_get_uri(self, object_store: ObjectStore):
        uri = object_store.get_uri("tmpfile_object_name")
        if isinstance(object_store, S3ObjectStore):
            assert uri == "s3://my-bucket/tmpfile_object_name"
        elif isinstance(object_store, LibcloudObjectStore):
            assert uri == "local://./tmpfile_object_name"
        else:
            raise NotImplementedError(f"Object store {type(object_store)} not implemented.")

    def test_get_file_size(self, object_store: ObjectStore, dummy_obj: pathlib.Path):
        object_name = "tmpfile_object_name"
        object_store.upload_object(object_name, str(dummy_obj))
        assert object_store.get_object_size(object_name) == dummy_obj.stat().st_size

    def test_get_file_size_not_found(self, object_store: ObjectStore):
        with pytest.raises(FileNotFoundError):
            object_store.get_object_size("not found object")

    def test_download(self, object_store: ObjectStore, dummy_obj: pathlib.Path, tmp_path: pathlib.Path):
        object_name = "tmpfile_object_name"
        object_store.upload_object(object_name, str(dummy_obj))
        filepath = str(tmp_path / "destination_path")
        cb = MockCallback(dummy_obj.stat().st_size)
        object_store.download_object(object_name, filepath, callback=cb)
        cb.assert_all_data_transferred()

    def test_download_not_found(self, object_store: ObjectStore):
        with pytest.raises(FileNotFoundError):
            object_store.download_object("not_found_object", filename="not used")
