# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import contextlib
import os
import pathlib
import random
import shutil
import time
from typing import Callable, Optional, Union

import pytest

from composer.core.event import Event
from composer.core.state import State
from composer.loggers import Logger, LogLevel
from composer.loggers.object_store_logger import ObjectStoreLogger
from composer.utils.object_store.object_store import ObjectStore


def my_filter_func(state: State, log_level: LogLevel, artifact_name: str):
    del state, log_level, artifact_name  # unused
    return False


class DummyObjectStore(ObjectStore):
    """Dummy ObjectStore implementation that is backed by a local directory."""

    def __init__(self, dir: pathlib.Path, always_fail: bool = False) -> None:
        self.dir = str(dir)
        self.always_fail = always_fail
        os.makedirs(self.dir, exist_ok=True)

    def get_uri(self, object_name: str) -> str:
        return 'local://' + object_name

    def _get_abs_path(self, object_name: str):
        return self.dir + '/' + object_name

    def upload_object(
        self,
        object_name: str,
        filename: Union[str, pathlib.Path],
        callback: Optional[Callable[[int, int], None]] = None,
    ) -> None:
        if self.always_fail and object_name != '.credentials_validated_successfully':
            raise RuntimeError
        time.sleep(random.random() * 0.5)  # random sleep to simulate random network latency
        shutil.copy2(filename, self._get_abs_path(object_name))

    def download_object(self,
                        object_name: str,
                        filename: Union[str, pathlib.Path],
                        overwrite: bool = False,
                        callback: Optional[Callable[[int, int], None]] = None) -> None:
        if self.always_fail:
            raise RuntimeError
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
    should_filter: bool = False,
):
    remote_dir = str(tmp_path / 'object_store')
    os.makedirs(remote_dir, exist_ok=True)

    object_store_logger = ObjectStoreLogger(
        object_store_cls=DummyObjectStore,
        object_store_kwargs={
            'dir': remote_dir,
        },
        num_concurrent_uploads=1,
        use_procs=use_procs,
        should_log_artifact=my_filter_func if should_filter else None,
        upload_staging_folder=str(tmp_path / 'staging_folder'),
        num_attempts=1,
    )
    logger = Logger(dummy_state, destinations=[object_store_logger])
    artifact_name = 'artifact_name'

    object_store_logger.run_event(Event.INIT, dummy_state, logger)

    file_path = os.path.join(tmp_path, f'file')
    with open(file_path, 'w+') as f:
        f.write('1')
    logger.file_artifact(LogLevel.FIT, artifact_name, file_path, overwrite=overwrite)

    file_path_2 = os.path.join(tmp_path, f'file_2')
    with open(file_path_2, 'w+') as f:
        f.write('2')

    post_close_ctx = contextlib.nullcontext()

    if not overwrite:
        # If not `overwrite_delay`, then the `logger.file_artifact` will raise a FileExistsException, because the upload will not even be enqueued
        # Otherwise, with a sufficient will be uploaded, and cleared from the queue, causing a runtime error to be raised on Event.BATCH_END or Event.EPOCH_END
        # A 2 second sleep should be enough here -- the DummyObjectStore will block for at most 0.5 seconds, and the ObjectStoreLogger polls every 0.1 seconds
        if overwrite_delay:
            post_close_ctx = pytest.warns(
                RuntimeWarning,
                match=r'The following objects may not have been uploaded, likely due to a worker crash: artifact_name')
            # Wait for the first upload to go through
            time.sleep(2)
            # Do the second upload -- it should enqueue
            logger.file_artifact(LogLevel.FIT, artifact_name, file_path_2, overwrite=overwrite)
            # Give it some time to finish the second upload
            # (Since the upload is really a file copy, it should be fast)
            time.sleep(2)
            # Then, crashes are detected on the next batch end / epoch end event
            with pytest.raises(RuntimeError):
                object_store_logger.run_event(Event.BATCH_END, dummy_state, logger)

            with pytest.raises(RuntimeError):
                object_store_logger.run_event(Event.EPOCH_END, dummy_state, logger)
        else:
            # Otherwise, if no delay, it should error when being enqueued
            with pytest.raises(FileExistsError):
                logger.file_artifact(LogLevel.FIT, artifact_name, file_path_2, overwrite=overwrite)

    object_store_logger.close(dummy_state, logger)

    with post_close_ctx:
        object_store_logger.post_close()

    # verify upload uri for artifact is correct
    upload_uri = object_store_logger.get_uri_for_artifact(artifact_name)
    expected_upload_uri = f'local://{artifact_name}'
    assert upload_uri == expected_upload_uri

    if not should_filter:
        # Test downloading artifact
        download_path = os.path.join(tmp_path, 'download')
        object_store_logger.get_file_artifact(artifact_name, download_path)
        with open(download_path, 'r') as f:
            assert f.read() == '1' if not overwrite else '2'

    # now assert that we have a dummy file in the artifact folder
    artifact_file = os.path.join(str(remote_dir), artifact_name)
    if should_filter:
        # If the filter works, nothing should be logged
        assert not os.path.exists(artifact_file)
    else:
        # Verify artifact contains the correct value
        with open(artifact_file, 'r') as f:
            assert f.read() == '1' if not overwrite else '2'


def test_object_store_logger(tmp_path: pathlib.Path, dummy_state: State):
    object_store_test_helper(tmp_path=tmp_path, dummy_state=dummy_state, use_procs=False)


def test_object_store_logger_use_procs(tmp_path: pathlib.Path, dummy_state: State):
    object_store_test_helper(tmp_path=tmp_path, dummy_state=dummy_state, use_procs=True)


@pytest.mark.filterwarnings(r'ignore:((.|\n)*)FileExistsError((.|\n)*):pytest.PytestUnhandledThreadExceptionWarning')
@pytest.mark.parametrize('overwrite_delay', [True, False])
def test_object_store_logger_no_overwrite(tmp_path: pathlib.Path, dummy_state: State, overwrite_delay: bool):
    object_store_test_helper(tmp_path=tmp_path,
                             dummy_state=dummy_state,
                             overwrite=False,
                             overwrite_delay=overwrite_delay)


def test_object_store_logger_should_log_artifact_filter(tmp_path: pathlib.Path, dummy_state: State):
    object_store_test_helper(tmp_path=tmp_path, dummy_state=dummy_state, should_filter=True)


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

    # Create the object store logger
    object_store_logger = ObjectStoreLogger(
        object_store_cls=DummyObjectStore,
        object_store_kwargs={
            'dir': tmp_path / 'object_store_backend',
        },
        num_concurrent_uploads=4,
        use_procs=use_procs,
        upload_staging_folder=str(tmp_path / 'staging_folder'),
        num_attempts=1,
    )

    logger = Logger(dummy_state, destinations=[object_store_logger])

    object_store_logger.run_event(Event.INIT, dummy_state, logger)

    # Queue the files for upload in rapid succession to the same artifact_name
    artifact_name = 'artifact_name'
    for i in range(num_files):
        file_path = tmp_path / 'samples' / f'sample_{i}'
        object_store_logger.log_file_artifact(dummy_state, LogLevel.FIT, artifact_name, file_path, overwrite=True)

    # Shutdown the logger. This should wait until all objects are uploaded
    object_store_logger.close(dummy_state, logger=logger)
    object_store_logger.post_close()

    # Assert that the artifact called "artifact_name" has the content of the last file uploaded file -- i.e. `num_files` - 1
    destination = tmp_path / 'downloaded_artifact'
    object_store_logger.get_file_artifact(artifact_name, str(destination), overwrite=False, progress_bar=False)
    with open(destination, 'r') as f:
        assert f.read() == str(num_files - 1)


@pytest.mark.filterwarnings(r'ignore:Exception in thread:pytest.PytestUnhandledThreadExceptionWarning')
def test_close_on_failure(tmp_path: pathlib.Path, dummy_state: State):
    """Test that .close() and .post_close() does not hang even when a worker crashes."""
    # Create the object store logger
    object_store_logger = ObjectStoreLogger(
        object_store_cls=DummyObjectStore,
        object_store_kwargs={
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

    logger = Logger(dummy_state, destinations=[object_store_logger])

    object_store_logger.run_event(Event.INIT, dummy_state, logger)

    logger.file_artifact(LogLevel.FIT, 'dummy_artifact_name', tmpfile_path)

    # Wait enough time for the file to be enqueued
    time.sleep(0.5)

    # Assert that the worker crashed
    with pytest.raises(RuntimeError):
        object_store_logger.run_event(Event.EPOCH_END, dummy_state, logger)

    # Enqueue the file again to ensure that the buffers are dirty
    logger.file_artifact(LogLevel.FIT, 'dummy_artifact_name', tmpfile_path)

    # Shutdown the logger. This should not hang or cause any exception
    object_store_logger.close(dummy_state, logger=logger)
    with pytest.warns(
            RuntimeWarning,
            match=r'The following objects may not have been uploaded, likely due to a worker crash: dummy_artifact_name'
    ):
        object_store_logger.post_close()
