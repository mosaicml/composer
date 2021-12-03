# Copyright 2021 MosaicML. All Rights Reserved.

import logging
import os
import shutil

log = logging.getLogger(__name__)

_RUN_DIRECTORY_KEY = "RUN_DIRECTORY"


def _get_run_directory():
    return os.environ.get(_RUN_DIRECTORY_KEY)


def get_run_directory():
    """Returns the run directory, if set, or None otherwise."""
    return _get_run_directory()


def get_relative_to_run_directory(path: str, base: str = ".") -> str:
    run_directory = get_run_directory()
    if run_directory is None:
        return os.path.join(base, path)
    return os.path.join(run_directory, path)


def clear_run_directory():
    if _RUN_DIRECTORY_KEY in os.environ:
        shutil.rmtree(os.environ[_RUN_DIRECTORY_KEY], ignore_errors=True)
