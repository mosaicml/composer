# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import contextlib
import copy
import pathlib
from typing import Type

import pytest

from composer.utils.object_store import LibcloudObjectStore, ObjectStore, S3ObjectStore, SFTPObjectStore
from composer.utils.object_store.sftp_object_store import SFTPObjectStore
from tests.utils.object_store.object_store_settings import get_object_store_ctx, object_store_kwargs, object_stores


class MockCallback:

    def __init__(self, total_num_bytes: int) -> None:
        self.total_num_bytes = total_num_bytes
        self.transferred_bytes = 0
        self.num_calls = 0

    def __call__(self, transferred: int, total: int):
        self.num_calls += 1
        assert transferred == 0 or transferred >= self.transferred_bytes, 'transferred should be monotonically increasing'
        self.transferred_bytes = transferred
        assert total == self.total_num_bytes

    def assert_all_data_transferred(self):
        assert self.total_num_bytes == self.transferred_bytes


@pytest.mark.parametrize('object_store_cls', object_stores)
class TestObjectStore:

    @pytest.fixture
    def object_store(self, object_store_cls: Type[ObjectStore], monkeypatch: pytest.MonkeyPatch,
                     tmp_path: pathlib.Path):
        with get_object_store_ctx(object_store_cls, monkeypatch, tmp_path):
            copied_config = copy.deepcopy(object_store_kwargs[object_store_cls])
            # type error: Type[ObjectStore] is not callable
            object_store = object_store_cls(**copied_config)  # type: ignore
            with object_store:
                yield object_store

    @pytest.fixture
    def dummy_obj(self, tmp_path: pathlib.Path):
        tmpfile_path = tmp_path / 'file_to_upload'
        with open(tmpfile_path, 'w+') as f:
            f.write('dummy content')
        return tmpfile_path

    def test_upload(self, object_store: ObjectStore, dummy_obj: pathlib.Path):
        object_name = 'tmpfile_object_name'
        cb = MockCallback(dummy_obj.stat().st_size)
        object_store.upload_object(object_name, str(dummy_obj), callback=cb)
        cb.assert_all_data_transferred()

    def test_get_uri(self, object_store: ObjectStore):
        uri = object_store.get_uri('tmpfile_object_name')
        if isinstance(object_store, S3ObjectStore):
            assert uri == 's3://my-bucket/folder/subfolder/tmpfile_object_name'
        elif isinstance(object_store, LibcloudObjectStore):
            assert uri == 'local://./tmpfile_object_name'
        elif isinstance(object_store, SFTPObjectStore):
            assert uri == 'sftp://test_user@localhost:23/tmpfile_object_name'
        else:
            raise NotImplementedError(f'Object store {type(object_store)} not implemented.')

    def test_get_file_size(self, object_store: ObjectStore, dummy_obj: pathlib.Path):
        object_name = 'tmpfile_object_name'
        object_store.upload_object(object_name, str(dummy_obj))
        assert object_store.get_object_size(object_name) == dummy_obj.stat().st_size

    def test_get_file_size_not_found(self, object_store: ObjectStore):
        with pytest.raises(FileNotFoundError):
            object_store.get_object_size('not found object')

    @pytest.mark.parametrize('overwrite', [True, False])
    def test_download(self, object_store: ObjectStore, dummy_obj: pathlib.Path, tmp_path: pathlib.Path,
                      overwrite: bool):
        object_name = 'tmpfile_object_name'
        object_store.upload_object(object_name, str(dummy_obj))
        filepath = str(tmp_path / 'destination_path')
        cb = MockCallback(dummy_obj.stat().st_size)
        object_store.download_object(object_name, filepath, callback=cb)
        ctx = contextlib.nullcontext() if overwrite else pytest.raises(FileExistsError)
        with ctx:
            object_store.download_object(object_name, filepath, callback=cb, overwrite=overwrite)
        cb.assert_all_data_transferred()

    def test_download_not_found(self, object_store: ObjectStore):
        with pytest.raises(FileNotFoundError):
            object_store.download_object('not_found_object', filename='not used')
