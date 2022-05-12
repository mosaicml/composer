# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import os
import pathlib
import time

import pytest

from composer.core.event import Event
from composer.core.state import State
from composer.loggers import Logger, LogLevel, ObjectStoreLoggerHparams
from composer.utils.object_store import ObjectStoreHparams


def my_filter_func(state: State, log_level: LogLevel, artifact_name: str):
    del state, log_level, artifact_name  # unused
    return False


def object_store_test_helper(tmpdir: pathlib.Path,
                             dummy_state: State,
                             monkeypatch: pytest.MonkeyPatch,
                             use_procs: bool = False,
                             overwrite: bool = True,
                             should_filter: bool = False):
    remote_dir = str(tmpdir / "object_store")

    os.makedirs(remote_dir, exist_ok=True)
    monkeypatch.setenv("OBJECT_STORE_KEY", remote_dir)  # for the local option, the key is the path
    provider = "local"
    container = "."
    hparams = ObjectStoreLoggerHparams(
        object_store_hparams=ObjectStoreHparams(
            provider=provider,
            container=container,
            key_environ="OBJECT_STORE_KEY",
        ),
        num_concurrent_uploads=1,
        use_procs=use_procs,
    )

    destination = hparams.initialize_object()
    if should_filter:
        destination.should_log_artifact = my_filter_func
    logger = Logger(dummy_state, [destination])
    artifact_name = "artifact_name"
    symlink_artifact_name = "latest_artifact"
    destination.run_event(Event.INIT, dummy_state, logger)

    file_path = os.path.join(tmpdir, f"file")
    with open(file_path, "w+") as f:
        f.write("1")
    logger.file_artifact(LogLevel.FIT, artifact_name, file_path, overwrite=overwrite)

    # Add symlink to ``artifact_name``.
    logger.symlink_artifact(LogLevel.FIT, artifact_name, symlink_artifact_name, overwrite=overwrite)

    file_path_2 = os.path.join(tmpdir, f"file_2")
    with open(file_path_2, "w+") as f:
        f.write("2")

    # If overwrite is False, then the worker will crash.
    logger.file_artifact(LogLevel.FIT, artifact_name, file_path_2, overwrite=overwrite)

    if not overwrite:
        # Give it some time to attempt the upload
        # (Since the upload is really a file copy, it should be fast)
        time.sleep(2)
        # Then, crashes are detected on the next batch end / epoch end event
        with pytest.raises(RuntimeError):
            destination.run_event(Event.BATCH_END, dummy_state, logger)

        with pytest.raises(RuntimeError):
            destination.run_event(Event.EPOCH_END, dummy_state, logger)

    destination.close(dummy_state, logger)
    destination.post_close()

    # verify upload uri for artifact is correct
    upload_uri = destination.get_uri_for_artifact(artifact_name)
    expected_upload_uri = f"{provider}://{container}/{artifact_name}"
    assert upload_uri == expected_upload_uri

    # verify upload uri for symlink is correct
    upload_uri = destination.get_uri_for_artifact(symlink_artifact_name)
    expected_upload_uri = f"{provider}://{container}/{symlink_artifact_name}"
    assert upload_uri == expected_upload_uri

    if not should_filter:
        # Test downloading artifact
        download_path = os.path.join(tmpdir, "download")
        destination.get_file_artifact(artifact_name, download_path)
        with open(download_path, "r") as f:
            assert f.read() == "1" if not overwrite else "2"

    # now assert that we have a dummy file in the artifact folder
    artifact_file = os.path.join(str(remote_dir), artifact_name)
    symlink_artifact_name = os.path.join(str(remote_dir), symlink_artifact_name + ".symlink")
    if should_filter:
        # If the filter works, nothing should be logged
        assert not os.path.exists(artifact_file)
        assert not os.path.exists(symlink_artifact_name)
    else:
        # Verify artifact contains the correct value
        with open(artifact_file, "r") as f:
            assert f.read() == "1" if not overwrite else "2"
        # Verify symlink is pointing to the artifact
        with open(symlink_artifact_name) as f:
            assert f.read() == artifact_name


@pytest.mark.timeout(15)
def test_object_store_logger_use_procs(tmpdir: pathlib.Path, dummy_state: State, monkeypatch: pytest.MonkeyPatch):
    object_store_test_helper(tmpdir=tmpdir, dummy_state=dummy_state, monkeypatch=monkeypatch, use_procs=True)


@pytest.mark.timeout(15)
@pytest.mark.filterwarnings(r"ignore:((.|\n)*)FileExistsError((.|\n)*):pytest.PytestUnhandledThreadExceptionWarning")
def test_object_store_logger_no_overwrite(tmpdir: pathlib.Path, dummy_state: State, monkeypatch: pytest.MonkeyPatch):
    object_store_test_helper(tmpdir=tmpdir, dummy_state=dummy_state, monkeypatch=monkeypatch, overwrite=False)


def test_object_store_logger_should_log_artifact_filter(tmpdir: pathlib.Path, dummy_state: State,
                                                        monkeypatch: pytest.MonkeyPatch):
    object_store_test_helper(tmpdir=tmpdir, dummy_state=dummy_state, monkeypatch=monkeypatch, should_filter=True)
