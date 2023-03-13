# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import contextlib
import multiprocessing
import os
import pathlib
import random
import shutil
import time
from typing import Any, Callable, Dict, Optional, Union
from unittest.mock import patch

import pytest

from composer.core import Event, State
from composer.loggers import Logger, RemoteUploaderDownloader
from composer.utils.object_store.object_store import ObjectStore


class DummyObjectStore(ObjectStore):
    """Dummy ObjectStore implementation that is backed by a local directory."""

    def __init__(self, dir: Optional[pathlib.Path] = None, always_fail: bool = False, **kwargs: Dict[str, Any]) -> None:
        self.dir = str(dir) if dir is not None else kwargs['bucket']
        self.always_fail = always_fail
        assert isinstance(self.dir, str)
        os.makedirs(self.dir, exist_ok=True)

    def get_uri(self, object_name: str) -> str:
        return 'local://' + object_name

    def _get_abs_path(self, object_name: str):
        assert isinstance(self.dir, str)
        return os.path.abspath(self.dir + '/' + object_name)

    def upload_object(
        self,
        object_name: str,
        filename: Union[str, pathlib.Path],
        callback: Optional[Callable[[int, int], None]] = None,
    ) -> None:
        if self.always_fail and object_name != '.credentials_validated_successfully':
            raise RuntimeError('Crash because you set always_fail to true!')
        time.sleep(random.random() * 0.5)  # random sleep to simulate random network latency
        shutil.copy2(filename, self._get_abs_path(object_name))

    def download_object(self,
                        object_name: str,
                        filename: Union[str, pathlib.Path],
                        overwrite: bool = False,
                        callback: Optional[Callable[[int, int], None]] = None) -> None:
        if self.always_fail:
            raise RuntimeError('Crash because you set always_fail to true!')
        if not overwrite and os.path.exists(filename):
            raise FileExistsError
        return shutil.copy2(self._get_abs_path(object_name), filename)

    def get_object_size(self, object_name: str) -> int:
        size = os.stat(self._get_abs_path(object_name)).st_size
        return size


def object_store_test_helper(
    tmp_path: pathlib.Path,
    dummy_state: State,
    use_procs: bool = False,
    overwrite: bool = True,
    overwrite_delay: bool = False,
    event_to_test: Event = Event.BATCH_END,
):
    remote_dir = str(tmp_path / 'object_store')
    os.makedirs(remote_dir, exist_ok=True)

    # Patching does not work when using multiprocessing with spawn, so we also
    # patch to use fork
    fork_context = multiprocessing.get_context('fork')
    with patch('composer.loggers.remote_uploader_downloader.S3ObjectStore', DummyObjectStore):
        with patch('composer.loggers.remote_uploader_downloader.multiprocessing.get_context', lambda _: fork_context):
            remote_uploader_downloader = RemoteUploaderDownloader(
                bucket_uri='s3://{remote_dir}',
                backend_kwargs={
                    'dir': remote_dir,
                },
                num_concurrent_uploads=1,
                use_procs=use_procs,
                upload_staging_folder=str(tmp_path / 'staging_folder'),
                num_attempts=1,
            )
            logger = Logger(dummy_state, destinations=[remote_uploader_downloader])
            remote_file_name = 'remote_file_name'

            remote_uploader_downloader.run_event(Event.INIT, dummy_state, logger)

            file_path = os.path.join(tmp_path, f'file')
            with open(file_path, 'w+') as f:
                f.write('1')
            logger.upload_file(remote_file_name, file_path, overwrite=overwrite)

            file_path_2 = os.path.join(tmp_path, f'file_2')
            with open(file_path_2, 'w+') as f:
                f.write('2')

            post_close_ctx = contextlib.nullcontext()

            if not overwrite:
                # If not `overwrite_delay`, then the `logger.upload_file` will raise a FileExistsException, because the upload will not even be enqueued
                # Otherwise, with a sufficient will be uploaded, and cleared from the queue, causing a runtime error to be raised on Event.BATCH_END or Event.EPOCH_END
                # A 2 second sleep should be enough here -- the DummyObjectStore will block for at most 0.5 seconds, and the RemoteUploaderDownloader polls every 0.1 seconds
                if overwrite_delay:
                    post_close_ctx = pytest.warns(
                        RuntimeWarning,
                        match=
                        r'The following objects may not have been uploaded, likely due to a worker crash: remote_file_name'
                    )
                    # Wait for the first upload to go through
                    time.sleep(2)
                    # Do the second upload -- it should enqueue
                    logger.upload_file(remote_file_name, file_path_2, overwrite=overwrite)
                    # Give it some time to finish the second upload
                    # (Since the upload is really a file copy, it should be fast)
                    time.sleep(2)
                    # Then, crashes are detected on the next batch end, but
                    # should be a FileExistsError not a runtime error because the parent will raise
                    # the fatal exception that the worker throws.
                    with pytest.raises(
                            FileExistsError,
                            match=
                            f'Object local://{remote_file_name} already exists, but allow_overwrite was set to False.'):
                        remote_uploader_downloader.run_event(event_to_test, dummy_state, logger)

                else:
                    # Otherwise, if no delay, it should error when being enqueued
                    with pytest.raises(
                            FileExistsError,
                            match=f'Object {remote_file_name} was already enqueued to be uploaded, but overwrite=False.'
                    ):
                        logger.upload_file(remote_file_name, file_path_2, overwrite=overwrite)

            remote_uploader_downloader.close(dummy_state, logger)

            with post_close_ctx:
                remote_uploader_downloader.post_close()

            # verify upload uri for file is correct
            upload_uri = remote_uploader_downloader.get_uri_for_file(remote_file_name)
            expected_upload_uri = f'local://{remote_file_name}'
            assert upload_uri == expected_upload_uri

            # Test downloading file
            download_path = os.path.join(tmp_path, 'download')
            remote_uploader_downloader.download_file(remote_file_name, download_path)
            with open(download_path, 'r') as f:
                assert f.read() == '1' if not overwrite else '2'

            # now assert that we have a dummy file in the remote folder
            remote_file = os.path.join(str(remote_dir), remote_file_name)
            # Verify file contains the correct value
            with open(remote_file, 'r') as f:
                assert f.read() == '1' if not overwrite else '2'


def test_remote_uploader_downloader(tmp_path: pathlib.Path, dummy_state: State):
    object_store_test_helper(tmp_path=tmp_path, dummy_state=dummy_state, use_procs=False)


def test_remote_uploader_downloader_use_procs(tmp_path: pathlib.Path, dummy_state: State):
    object_store_test_helper(tmp_path=tmp_path, dummy_state=dummy_state, use_procs=True)


@pytest.mark.filterwarnings(r'ignore:((.|\n)*)FileExistsError((.|\n)*):pytest.PytestUnhandledThreadExceptionWarning')
@pytest.mark.parametrize('overwrite_delay', [True, False])
@pytest.mark.parametrize('event_to_test', [Event.BATCH_END, Event.EPOCH_END])
def test_remote_uploader_downloader_no_overwrite(tmp_path: pathlib.Path, dummy_state: State, overwrite_delay: bool,
                                                 event_to_test: Event):
    if not overwrite_delay and event_to_test == Event.EPOCH_END:
        pytest.skip('event_to_test does not affect the overwrite_delay=False part of the test')
    object_store_test_helper(tmp_path=tmp_path,
                             dummy_state=dummy_state,
                             overwrite=False,
                             overwrite_delay=overwrite_delay,
                             event_to_test=event_to_test)


@pytest.mark.parametrize('use_procs', [True, False])
def test_race_with_overwrite(tmp_path: pathlib.Path, use_procs: bool, dummy_state: State):
    # Test a race condition with the object store logger where multiple files with the same name are logged in rapid succession
    # The latest version should be the one that is uploaded

    # Setup: Prep files
    num_files = 10
    os.makedirs(tmp_path / 'samples')
    for i in range(num_files):
        with open(tmp_path / 'samples' / f'sample_{i}', 'w+') as f:
            f.write(str(i))

    # Patching does not work when using multiprocessing with spawn, so we also
    # patch to use fork
    fork_context = multiprocessing.get_context('fork')
    with patch('composer.loggers.remote_uploader_downloader.S3ObjectStore', DummyObjectStore):
        with patch('composer.loggers.remote_uploader_downloader.multiprocessing.get_context', lambda _: fork_context):
            # Create the object store logger
            remote_uploader_downloader = RemoteUploaderDownloader(
                bucket_uri=f"s3://{tmp_path}/'object_store_backend",
                backend_kwargs={
                    'dir': tmp_path / 'object_store_backend',
                },
                num_concurrent_uploads=4,
                use_procs=use_procs,
                upload_staging_folder=str(tmp_path / 'staging_folder'),
                num_attempts=1,
            )

            logger = Logger(dummy_state, destinations=[remote_uploader_downloader])

            remote_uploader_downloader.run_event(Event.INIT, dummy_state, logger)

            # Queue the files for upload in rapid succession to the same remote_file_name
            remote_file_name = 'remote_file_name'
            for i in range(num_files):
                file_path = tmp_path / 'samples' / f'sample_{i}'
                remote_uploader_downloader.upload_file(dummy_state, remote_file_name, file_path, overwrite=True)

            # Shutdown the logger. This should wait until all objects are uploaded
            remote_uploader_downloader.close(dummy_state, logger=logger)
            remote_uploader_downloader.post_close()

            # Assert that the file called "remote_file_name" has the content of the last file uploaded file -- i.e. `num_files` - 1
            destination = tmp_path / 'downloaded_file'
            remote_uploader_downloader.download_file(remote_file_name,
                                                     str(destination),
                                                     overwrite=False,
                                                     progress_bar=False)
            with open(destination, 'r') as f:
                assert f.read() == str(num_files - 1)


@pytest.mark.filterwarnings(r'ignore:Exception in thread:pytest.PytestUnhandledThreadExceptionWarning')
def test_close_on_failure(tmp_path: pathlib.Path, dummy_state: State):
    """Test that .close() and .post_close() does not hang even when a worker crashes."""

    with patch('composer.loggers.remote_uploader_downloader.S3ObjectStore', DummyObjectStore):
        # Create the object store logger
        remote_uploader_downloader = RemoteUploaderDownloader(
            bucket_uri=f"s3://{tmp_path}/'object_store_backend",
            backend_kwargs={
                'dir': tmp_path / 'object_store_backend',
                'always_fail': True,
            },
            num_concurrent_uploads=1,
            use_procs=False,
            upload_staging_folder=str(tmp_path / 'staging_folder'),
            num_attempts=1,
        )

        # Enqueue a file. Because `always_fail` is True, it will cause the worker to crash

        tmpfile_path = tmp_path / 'dummy_file'
        with open(tmpfile_path, 'w+') as f:
            f.write('hi')

        logger = Logger(dummy_state, destinations=[remote_uploader_downloader])

        remote_uploader_downloader.run_event(Event.INIT, dummy_state, logger)

        logger.upload_file('dummy_remote_file_name', tmpfile_path)

        # Wait enough time for the file to be enqueued
        time.sleep(0.5)

        # Assert that the worker crashed
        with pytest.raises(RuntimeError):
            remote_uploader_downloader.run_event(Event.EPOCH_END, dummy_state, logger)

        # Enqueue the file again to ensure that the buffers are dirty
        logger.upload_file('dummy_remote_file_name', tmpfile_path)

        # Shutdown the logger. This should not hang or cause any exception
        remote_uploader_downloader.close(dummy_state, logger=logger)
        with pytest.warns(
                RuntimeWarning,
                match=
                r'The following objects may not have been uploaded, likely due to a worker crash: dummy_remote_file_name'
        ):
            remote_uploader_downloader.post_close()


def test_valid_backend_names():
    valid_backend_names = ['s3', 'libcloud', 'sftp']
    with patch('composer.loggers.remote_uploader_downloader.S3ObjectStore') as _, \
         patch('composer.loggers.remote_uploader_downloader.SFTPObjectStore') as _, \
         patch('composer.loggers.remote_uploader_downloader.LibcloudObjectStore') as _:
        for name in valid_backend_names:
            remote_uploader_downloader = RemoteUploaderDownloader(bucket_uri=f'{name}://not-a-real-bucket')
            # Access the remote_backend property so that it is built
            _ = remote_uploader_downloader.remote_backend

    with pytest.raises(ValueError):
        remote_uploader_downloader = RemoteUploaderDownloader(bucket_uri='magicloud://not-a-real-bucket')
        # Access the remote_backend property so that it is built
        _ = remote_uploader_downloader.remote_backend


# We put this filter here because when the worker raises an exception, pytest throws a warning which fails the test.
@pytest.mark.filterwarnings(r'ignore:Exception in thread:pytest.PytestUnhandledThreadExceptionWarning')
def test_exception_queue_works(tmp_path: pathlib.Path, dummy_state: State):
    """Test that exceptions get put on the exception queue and get thrown"""

    with patch('composer.loggers.remote_uploader_downloader.S3ObjectStore', DummyObjectStore):
        # Create the object store logger
        remote_uploader_downloader = RemoteUploaderDownloader(
            bucket_uri=f"s3://{tmp_path}/'object_store_backend",
            backend_kwargs={
                'dir': tmp_path / 'object_store_backend',
                'always_fail': True,
            },
            num_concurrent_uploads=1,
            use_procs=False,
            upload_staging_folder=str(tmp_path / 'staging_folder'),
            num_attempts=1,
        )

        # Enqueue a file. Because `always_fail` is True, it will cause the worker to crash

        tmpfile_path = tmp_path / 'dummy_file'
        with open(tmpfile_path, 'w+') as f:
            f.write('hi')

        logger = Logger(dummy_state, destinations=[remote_uploader_downloader])

        remote_uploader_downloader.run_event(Event.INIT, dummy_state, logger)

        logger.upload_file('dummy_remote_file_name', tmpfile_path)

        # Wait enough time for the file to be enqueued and the exception to be enqueued
        time.sleep(2.0)

        # Make sure the exception got enqueued.
        assert not remote_uploader_downloader._exception_queue.empty()

        # Assert that the worker crashed with the worker's error not the general
        # 'Upload worker crashed. Please check the logs.' error.
        with pytest.raises(RuntimeError, match='Crash because you set*'):
            remote_uploader_downloader.run_event(Event.EPOCH_END, dummy_state, logger)
