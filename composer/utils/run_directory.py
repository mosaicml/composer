# Copyright 2021 MosaicML. All Rights Reserved.

import logging
import os

log = logging.getLogger(__name__)

_RUN_DIRECTORY_KEY = "RUN_DIRECTORY"


def get_run_directory():
    return os.environ.get(_RUN_DIRECTORY_KEY)


def get_relative_to_run_directory(path: str, base: str = ".") -> str:
    run_directory = get_run_directory()
    if run_directory is None:
        return os.path.join(base, path)
    return os.path.join(run_directory, path)
