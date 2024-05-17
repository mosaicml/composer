# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import os
import pathlib
import shutil
import tempfile
import time
from typing import Any, Callable, Dict, Optional, Union
from unittest.mock import patch

import pytest

from composer.utils.object_store.object_store import ObjectStore
from composer.utils.remote_uploader import RemoteUploader


class DummyObjectStore(ObjectStore):
    """Dummy ObjectStore implementation that is backed by a local directory."""

    def __init__(self, **kwargs: Dict[str, Any]) -> None:
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.root = self.tmp_dir.name
        self.sleep_sec = 0
        self.raise_error = False
        self.dest_filename = ''

    def upload_object(
        self,
        object_name: str,
        filename: Union[str, pathlib.Path],
        callback: Optional[Callable[[int, int], None]] = None,
    ) -> None:
        if self.raise_error:
            raise RuntimeError('Raise Error intentionally')
        time.sleep(self.sleep_sec)
        dest_filename = pathlib.Path(self.root) / object_name
        shutil.copy2(filename, dest_filename)
        self.dest_filename = dest_filename

    def get_object_size(self, object_name: str) -> int:
        object_path = pathlib.Path(self.root) / object_name
        size = os.stat(object_path).st_size
        return size


def test_upload_mutliple_files():
    with patch('composer.utils.file_helpers.S3ObjectStore', DummyObjectStore):
        remote_uploader = RemoteUploader(remote_folder='S3://whatever/path',)

        tmp_dir = tempfile.TemporaryDirectory()
        tmp_path = tmp_dir.name
        # create source files
        files_num = 5
        for i in range(files_num):
            file_path = os.path.join(tmp_path, str(i))
            with open(file_path, 'w') as f:
                f.write(str(i))

        for i in range(files_num):
            remote_uploader.upload_file_async(
                remote_file_name=str(i),
                file_path=pathlib.Path(os.path.join(tmp_path, str(i))),
                overwrite=True,
            )
        remote_uploader.wait_and_close()

        # Check if the files exists in remote object store
        remote_path = remote_uploader.object_store.root  # pyright: ignore
        for i in range(5):
            remote_file_path = os.path.join(remote_path, str(i))
            with open(remote_file_path, 'r') as f:
                assert f.read() == str(i)


@pytest.mark.parametrize(
    'overwrite',
    [True, False],
)
def test_overwrite(overwrite: bool):
    with patch('composer.utils.file_helpers.S3ObjectStore', DummyObjectStore):
        remote_uploader = RemoteUploader(remote_folder='S3://whatever/path',)
        tmp_dir = tempfile.TemporaryDirectory()
        tmp_path = tmp_dir.name
        file_path = os.path.join(tmp_path, 'a')
        with open(file_path, 'w') as f:
            f.write('1')

        remote_uploader.upload_file_async(
            remote_file_name='a',
            file_path=pathlib.Path(os.path.join(tmp_path, 'a')),
            overwrite=True,
        ).result()
        remote_root_path = remote_uploader.object_store.root  # pyright: ignore
        if overwrite:
            file_path = os.path.join(tmp_path, 'a')
            with open(file_path, 'w') as f:
                f.write('2')
            remote_uploader.upload_file_async(
                remote_file_name='a',
                file_path=pathlib.Path(os.path.join(tmp_path, 'a')),
                overwrite=True,
            ).result()
            remote_file_path = os.path.join(remote_root_path, 'a')
            with open(remote_file_path, 'r') as f:
                assert f.read() == '2'
        else:
            with pytest.raises(FileExistsError):
                remote_uploader.upload_file_async(
                    remote_file_name='a',
                    file_path=pathlib.Path(os.path.join(tmp_path, 'a')),
                    overwrite=False,
                ).result()


def test_check_workers():
    with patch('composer.utils.file_helpers.S3ObjectStore', DummyObjectStore):
        remote_uploader = RemoteUploader(remote_folder='S3://whatever/path',)
        tmp_dir = tempfile.TemporaryDirectory()
        tmp_path = tmp_dir.name
        file_path = os.path.join(tmp_path, 'a')
        with open(file_path, 'w') as f:
            f.write('1')

        remote_uploader.object_store.raise_error = True  # pyright: ignore
        future = remote_uploader.upload_file_async(
            remote_file_name='a',
            file_path=pathlib.Path(os.path.join(tmp_path, 'a')),
            overwrite=True,
        )

        with pytest.raises(RuntimeError, match='Raise Error intentionally'):
            while not future.done():
                time.sleep(0.5)
            remote_uploader.check_workers()


def test_wait():
    with patch('composer.utils.file_helpers.S3ObjectStore', DummyObjectStore):
        remote_uploader = RemoteUploader(remote_folder='S3://whatever/path',)
        tmp_dir = tempfile.TemporaryDirectory()
        tmp_path = tmp_dir.name
        file_path = os.path.join(tmp_path, 'a')
        with open(file_path, 'w') as f:
            f.write('1')

        futures = []
        for _ in range(5):
            futures.append(
                remote_uploader.upload_file_async(
                    remote_file_name='a',
                    file_path=pathlib.Path(os.path.join(tmp_path, 'a')),
                    overwrite=True,
                ),
            )
        remote_uploader.wait()
        assert len(remote_uploader.futures) == 0
        for future in futures:
            assert future.done() == True


def test_wait_and_close():
    with patch('composer.utils.file_helpers.S3ObjectStore', DummyObjectStore):
        remote_uploader = RemoteUploader(remote_folder='S3://whatever/path',)
        tmp_dir = tempfile.TemporaryDirectory()
        tmp_path = tmp_dir.name
        file_path = os.path.join(tmp_path, 'a')
        with open(file_path, 'w') as f:
            f.write('1')

        futures = []
        for _ in range(5):
            futures.append(
                remote_uploader.upload_file_async(
                    remote_file_name='a',
                    file_path=pathlib.Path(os.path.join(tmp_path, 'a')),
                    overwrite=True,
                ),
            )
        remote_uploader.wait_and_close()
        for future in futures:
            assert future.done() == True
        assert len(remote_uploader.futures) == 0
