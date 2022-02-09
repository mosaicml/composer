# Copyright 2021 MosaicML. All Rights Reserved.

"""Utilities for working with the run directory.

TODO
"""
import datetime
import logging
import os
import pathlib
import time

from composer.utils import dist

log = logging.getLogger(__name__)

_RUN_DIRECTORY_KEY = "COMPOSER_RUN_DIRECTORY"

_start_time_str = datetime.datetime.now().isoformat()

__all__ = [
    "get_node_run_directory",
    "get_run_directory",
    "get_modified_files",
    "get_run_directory_timestamp",
]


def get_node_run_directory() -> str:
    """Returns the run directory for the node. This folder is shared by all ranks on the node.

    Returns:
        str: The node run directory.
    """
    node_run_directory = os.environ.get(_RUN_DIRECTORY_KEY, os.path.join("runs", _start_time_str))
    if node_run_directory.endswith(os.path.sep):
        # chop off the training slash so os.path.basename would work as expected
        node_run_directory = node_run_directory[:-1]
    os.makedirs(node_run_directory, exist_ok=True)
    return os.path.abspath(node_run_directory)


def get_run_directory() -> str:
    """Returns the run directory for the current rank.

    Returns:
        str: The run directory.
    """
    run_dir = os.path.join(get_node_run_directory(), f"rank_{dist.get_global_rank()}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def get_modified_files(modified_since_timestamp: float, *, ignore_hidden: bool = True):
    """Returns a list of files (recursively) in the run directory that have been modified since
    ``modified_since_timestamp``.

    Args:
        modified_since_timestamp (float): Minimum last modified timestamp(in seconds since EPOCH)
            of files to include.
        ignore_hidden (bool, optional): Whether to ignore hidden files and folders (default: ``True``)
    Returns:
        List[str]: List of filepaths that have been modified since ``modified_since_timestamp``
    """
    modified_files = []
    run_directory = get_run_directory()
    if run_directory is None:
        raise RuntimeError("Run directory is not defined")
    for root, dirs, files in os.walk(run_directory):
        del dirs  # unused
        for file in files:
            if ignore_hidden and any(x.startswith(".") for x in file.split(os.path.sep)):
                # skip hidden files and folders
                continue
            filepath = os.path.join(root, file)
            modified_time = os.path.getmtime(filepath)
            if modified_time >= modified_since_timestamp:
                modified_files.append(filepath)
    return modified_files


def get_run_directory_timestamp() -> float:
    """Returns the current timestamp on the run directory filesystem. Note that the disk time can differ from system
    time (e.g. when using network filesystems).

    Returns:
        float: The current timestamp on the run directory filesystem.
    """
    run_directory = get_run_directory()
    if run_directory is None:
        raise RuntimeError("Run directory is not defined")
    python_time = time.time()
    touch_file = (pathlib.Path(run_directory) / f".{python_time}")
    touch_file.touch()
    new_last_uploaded_timestamp = os.path.getmtime(str(touch_file))
    os.remove(str(touch_file))
    return new_last_uploaded_timestamp
