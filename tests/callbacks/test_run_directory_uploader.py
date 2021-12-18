# Copyright 2021 MosaicML. All Rights Reserved.

import os
import pathlib

import pytest

from composer.callbacks import RunDirectoryUploaderHparams
from composer.core.event import Event
from composer.core.logging import Logger
from composer.core.state import State
from composer.utils.run_directory import get_run_directory


# This test is flaky see: https://github.com/mosaicml/composer/issues/176
@pytest.mark.parametrize("use_procs", [False, True])
# TODO(ravi) -- remove the pytest.in #110. The TRAINING_END event is likely slow as it has to copy many
# files created by the ddp test. #110 grately reduces the number of files from the DDP test.
@pytest.mark.timeout(15)
@pytest.mark.xfail
def test_run_directory_uploader(tmpdir: pathlib.Path, use_procs: bool, dummy_state: State, dummy_logger: Logger):
    dummy_state.epoch = 0
    dummy_state.step = 0
    remote_dir = str(tmpdir / "run_directory_copy")
    os.makedirs(remote_dir, exist_ok=True)
    hparams = RunDirectoryUploaderHparams(
        provider='local',
        upload_every_n_batches=1,
        key=remote_dir,  # for the local option, the key is the path
        container=".",
        num_concurrent_uploads=1,
        use_procs=use_procs,
    )

    uploader = hparams.initialize_object()
    uploader.run_event(Event.INIT, dummy_state, dummy_logger)
    run_directory = get_run_directory()
    assert run_directory is not None
    with open(os.path.join(run_directory, "dummy_file"), "w+") as f:
        f.write("Hello, world!")
    uploader.run_event(Event.BATCH_END, dummy_state, dummy_logger)
    uploader.run_event(Event.TRAINING_END, dummy_state, dummy_logger)
    uploader.close()
    uploader.post_close()

    # now assert that we have a dummy file in the run directory copy folder
    with open(os.path.join(remote_dir, run_directory, "dummy_file"), "r") as f:
        assert f.read() == "Hello, world!"
