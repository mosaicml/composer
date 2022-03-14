# Copyright 2021 MosaicML. All Rights Reserved.

import os
import pathlib

import pytest

from composer.callbacks import RunDirectoryUploaderHparams
from composer.core.event import Event
from composer.core.state import State
from composer.loggers import Logger
from composer.utils import dist, run_directory


@pytest.mark.parametrize("use_procs", [False, True])
@pytest.mark.timeout(15)
def test_run_directory_uploader(tmpdir: pathlib.Path, use_procs: bool, dummy_state: State, dummy_logger: Logger,
                                monkeypatch: pytest.MonkeyPatch):
    remote_dir = str(tmpdir / "run_directory_copy")

    os.makedirs(remote_dir, exist_ok=True)
    monkeypatch.setenv("RUN_DIRECTORY_UPLOADER_KEY", remote_dir)  # for the local option, the key is the path
    provider = "local"
    container = "."
    hparams = RunDirectoryUploaderHparams(
        provider=provider,
        upload_every_n_batches=1,
        key_environ="RUN_DIRECTORY_UPLOADER_KEY",
        container=container,
        num_concurrent_uploads=1,
        use_procs=use_procs,
    )

    uploader = hparams.initialize_object()
    filename = "dummy_file"
    local_file = os.path.join(run_directory.get_run_directory(), filename)
    uploader.run_event(Event.INIT, dummy_state, dummy_logger)
    with open(local_file, "w+") as f:
        f.write("Hello, world!")
    uploader.run_event(Event.BATCH_END, dummy_state, dummy_logger)
    uploader.close()
    uploader.post_close()
    test_name = os.path.basename(tmpdir)

    # verify upload uri is correct
    upload_uri = uploader.get_uri_for_uploaded_file(local_file)
    expected_upload_uri = f"{provider}://{container}/{uploader._object_name_prefix}{filename}"
    assert upload_uri == expected_upload_uri

    # now assert that we have a dummy file in the run directory copy folder
    with open(os.path.join(remote_dir, test_name, f"rank_{dist.get_global_rank()}", "dummy_file"), "r") as f:
        assert f.read() == "Hello, world!"
