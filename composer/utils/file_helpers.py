# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Helpers for working with files."""

from __future__ import annotations

import os
import pathlib
import re
from typing import TYPE_CHECKING, Iterator, Optional, Union

import requests
import tqdm

from composer.core.time import Time, Timestamp
from composer.utils import dist
from composer.utils.iter_helpers import iterate_with_pbar
from composer.utils.object_store import ObjectStore

if TYPE_CHECKING:
    from composer.loggers import LoggerDestination

__all__ = [
    'GetFileNotFoundException',
    'get_file',
    'ensure_folder_is_empty',
    'ensure_folder_has_no_conflicting_files',
    'format_name_with_dist',
    'format_name_with_dist_and_time',
    'is_tar',
]


class GetFileNotFoundException(RuntimeError):
    """Exception if :meth:`get_file` failed due to a not found error."""
    pass


def is_tar(name: Union[str, pathlib.Path]) -> bool:
    """Returns whether ``name`` has a tar-like extension.

    Args:
        name (str | pathlib.Path): The name to check.

    Returns:
        bool: Whether ``name`` is a tarball.
    """
    return any(str(name).endswith(x) for x in (".tar", ".tgz", ".tar.gz", ".tar.bz2", ".tar.lzma"))


def ensure_folder_is_empty(folder_name: Union[str, pathlib.Path]):
    """Ensure that the given folder is empty.

    Hidden files and folders (those beginning with ``.``) and ignored. Sub-folders are checked recursively.

    Args:
        folder_name (str | pathlib.Path): The folder to ensure is empty.

    Raises:
        FileExistsError: If ``folder_name`` contains any non-hidden files, recursively.
    """
    for root, dirs, files in os.walk(folder_name, topdown=True):
        # Filter out hidden folders
        dirs[:] = (x for x in dirs if not x.startswith('.'))
        for file in files:
            if not file.startswith("."):
                raise FileExistsError(f"{folder_name} is not empty; {os.path.join(root, file)} exists.")


def ensure_folder_has_no_conflicting_files(folder_name: Union[str, pathlib.Path], filename: str, timestamp: Timestamp):
    """Ensure that the given folder does not have any files conflicting with the ``filename`` format string. If any
    filename is formatted with a timestamp where the epoch, batch, sample, or token counts are after ``timestamp``, a
    ``FileExistsError`` will be raised. ``filename`` and occurs later than ``timestamp``, raise a ``FileExistsError``.

    Args:
        folder_name (str | pathlib.Path): The folder to inspect.
        filename (str): The pattern string for potential files.
        timestamp (Timestamp): Ignore any files that occur before the provided timestamp.

    Raises:
        FileExistsError: If ``folder_name`` contains any files matching the ``filename`` template before ``timestamp``.
    """
    # Prepare regex pattern by replacing f-string formatting with regex.
    pattern = f"^{filename}$"
    # Format time vars for capture
    time_names = ["epoch", "batch", "sample", "token", "batch_in_epoch", "sample_in_epoch", "token_in_epoch"]
    captured_names = {time_name: f"{{{time_name}}}" in filename for time_name in time_names}
    for time_name, is_captured in captured_names.items():
        if is_captured:
            pattern = pattern.replace(f"{{{time_name}}}", f"(?P<{time_name}>\\d+)")
    # Format rank information
    pattern = pattern.format(rank=dist.get_global_rank(),
                             local_rank=dist.get_local_rank(),
                             world_size=dist.get_world_size(),
                             local_world_size=dist.get_local_world_size(),
                             node_rank=dist.get_node_rank())

    template = re.compile(pattern)

    for file in os.listdir(folder_name):
        match = template.match(file)
        # Encountered an invalid match
        if match is not None:
            valid_match = True
            # Check each base unit of time and flag later checkpoints
            if captured_names["token"] and Time.from_token(int(match.group("token"))) > timestamp.token:
                valid_match = False
            elif captured_names["sample"] and Time.from_sample(int(match.group("sample"))) > timestamp.sample:
                valid_match = False
            elif captured_names["batch"] and Time.from_batch(int(match.group("batch"))) > timestamp.batch:
                valid_match = False
            elif captured_names["epoch"] and Time.from_epoch(int(match.group("epoch"))) > timestamp.epoch:
                valid_match = False
            # If epoch count is same, check batch_in_epoch, sample_in_epoch, token_in_epoch
            elif captured_names["epoch"] and Time.from_epoch(int(match.group("epoch"))) == timestamp.epoch:
                if captured_names["token_in_epoch"] and Time.from_token(int(
                        match.group("token_in_epoch"))) > timestamp.token_in_epoch:
                    valid_match = False
                elif captured_names["sample_in_epoch"] and Time.from_sample(int(
                        match.group("sample_in_epoch"))) > timestamp.sample_in_epoch:
                    valid_match = False
                elif captured_names["batch_in_epoch"] and Time.from_batch(int(
                        match.group("batch_in_epoch"))) > timestamp.batch_in_epoch:
                    valid_match = False
            if not valid_match:
                raise FileExistsError(
                    f"{os.path.join(folder_name, file)} exists and conflicts in namespace with a future checkpoint of the current run."
                )


FORMAT_NAME_WITH_DIST_TABLE = """
+------------------------+-------------------------------------------------------+
| Variable               | Description                                           |
+========================+=======================================================+
| ``{run_name}``         | The name of the training run. See                     |
|                        | :attr:`~composer.loggers.logger.Logger.run_name`.     |
+------------------------+-------------------------------------------------------+
| ``{rank}``             | The global rank, as returned by                       |
|                        | :func:`~composer.utils.dist.get_global_rank`.         |
+------------------------+-------------------------------------------------------+
| ``{local_rank}``       | The local rank of the process, as returned by         |
|                        | :func:`~composer.utils.dist.get_local_rank`.          |
+------------------------+-------------------------------------------------------+
| ``{world_size}``       | The world size, as returned by                        |
|                        | :func:`~composer.utils.dist.get_world_size`.          |
+------------------------+-------------------------------------------------------+
| ``{local_world_size}`` | The local world size, as returned by                  |
|                        | :func:`~composer.utils.dist.get_local_world_size`.    |
+------------------------+-------------------------------------------------------+
| ``{node_rank}``        | The node rank, as returned by                         |
|                        | :func:`~composer.utils.dist.get_node_rank`.           |
+------------------------+-------------------------------------------------------+
"""


def format_name_with_dist(format_str: str, run_name: str, **extra_format_kwargs: object):
    formatted_str = format_str.format(
        run_name=run_name,
        rank=dist.get_global_rank(),
        local_rank=dist.get_local_rank(),
        world_size=dist.get_world_size(),
        local_world_size=dist.get_local_world_size(),
        node_rank=dist.get_node_rank(),
        **extra_format_kwargs,
    )
    return formatted_str


format_name_with_dist.__doc__ = f"""
Format ``format_str`` with the ``run_name``, distributed variables, and ``extra_format_kwargs``.

The following format variables are available:

{FORMAT_NAME_WITH_DIST_TABLE}

For example, assume that the rank is ``0``. Then:

>>> from composer.utils import format_name_with_dist
>>> format_str = '{{run_name}}/rank{{rank}}.{{extension}}'
>>> format_name_with_dist(
...     format_str,
...     run_name='awesome_training_run',
...     extension='json',
... )
'awesome_training_run/rank0.json'

Args:
    format_str (str): The format string for the checkpoint filename.
    run_name (str): The value for the ``{{run_name}}`` format variable.
    extra_format_kwargs (object): Any additional :meth:`~str.format` kwargs.
"""

FORMAT_NAME_WITH_DIST_AND_TIME_TABLE = """
+------------------------+--------------------------------------------------------+
| Variable               | Description                                            |
+========================+========================================================+
| ``{run_name}``         | The name of the training run. See                      |
|                        | :attr:`~composer.loggers.logger.Logger.run_name`.      |
+------------------------+--------------------------------------------------------+
| ``{rank}``             | The global rank, as returned by                        |
|                        | :func:`~composer.utils.dist.get_global_rank`.          |
+------------------------+--------------------------------------------------------+
| ``{local_rank}``       | The local rank of the process, as returned by          |
|                        | :func:`~composer.utils.dist.get_local_rank`.           |
+------------------------+--------------------------------------------------------+
| ``{world_size}``       | The world size, as returned by                         |
|                        | :func:`~composer.utils.dist.get_world_size`.           |
+------------------------+--------------------------------------------------------+
| ``{local_world_size}`` | The local world size, as returned by                   |
|                        | :func:`~composer.utils.dist.get_local_world_size`.     |
+------------------------+--------------------------------------------------------+
| ``{node_rank}``        | The node rank, as returned by                          |
|                        | :func:`~composer.utils.dist.get_node_rank`.            |
+------------------------+--------------------------------------------------------+
| ``{epoch}``            | The total epoch count, as returned by                  |
|                        | :meth:`~composer.core.time.Timestamp.epoch`.           |
+------------------------+--------------------------------------------------------+
| ``{batch}``            | The total batch count, as returned by                  |
|                        | :meth:`~composer.core.time.Timestamp.batch`.           |
+------------------------+--------------------------------------------------------+
| ``{batch_in_epoch}``   | The batch count in the current epoch, as returned by   |
|                        | :meth:`~composer.core.time.Timestamp.batch_in_epoch`.  |
+------------------------+--------------------------------------------------------+
| ``{sample}``           | The total sample count, as returned by                 |
|                        | :meth:`~composer.core.time.Timestamp.sample`.          |
+------------------------+--------------------------------------------------------+
| ``{sample_in_epoch}``  | The sample count in the current epoch, as returned by  |
|                        | :meth:`~composer.core.time.Timestamp.sample_in_epoch`. |
+------------------------+--------------------------------------------------------+
| ``{token}``            | The total token count, as returned by                  |
|                        | :meth:`~composer.core.time.Timestamp.token`.           |
+------------------------+--------------------------------------------------------+
| ``{token_in_epoch}``   | The token count in the current epoch, as returned by   |
|                        | :meth:`~composer.core.time.Timestamp.token_in_epoch`.  |
+------------------------+--------------------------------------------------------+
"""


def format_name_with_dist_and_time(format_str: str, run_name: str, timestamp: Timestamp, **extra_format_kwargs: object):
    formatted_str = format_str.format(
        run_name=run_name,
        rank=dist.get_global_rank(),
        local_rank=dist.get_local_rank(),
        world_size=dist.get_world_size(),
        local_world_size=dist.get_local_world_size(),
        node_rank=dist.get_node_rank(),
        epoch=int(timestamp.epoch),
        batch=int(timestamp.batch),
        batch_in_epoch=int(timestamp.batch_in_epoch),
        sample=int(timestamp.sample),
        sample_in_epoch=int(timestamp.sample_in_epoch),
        token=int(timestamp.token),
        token_in_epoch=int(timestamp.token_in_epoch),
        **extra_format_kwargs,
    )
    return formatted_str


format_name_with_dist_and_time.__doc__ = f"""\
Format ``format_str`` with the ``run_name``, distributed variables, ``timestamp``, and ``extra_format_kwargs``.

In addition to the variables specified via ``extra_format_kwargs``, the following format variables are available:

{FORMAT_NAME_WITH_DIST_AND_TIME_TABLE}

For example, assume that the current epoch is ``0``, batch is ``0``, and rank is ``0``. Then:

>>> from composer.utils import format_name_with_dist_and_time
>>> format_str = '{{run_name}}/ep{{epoch}}-ba{{batch}}-rank{{rank}}.{{extension}}'
>>> format_name_with_dist_and_time(
...     format_str,
...     run_name='awesome_training_run',
...     timestamp=state.timestamp,
...     extension='json',
... )
'awesome_training_run/ep0-ba0-rank0.json'

Args:
    format_str (str): The format string for the checkpoint filename.
    run_name (str): The value for the ``{{run_name}}`` format variable.
    timestamp (Timestamp): The timestamp.
    extra_format_kwargs (object): Any additional :meth:`~str.format` kwargs.
"""


def get_file(
    path: str,
    destination: str,
    object_store: Optional[Union[ObjectStore, LoggerDestination]] = None,
    chunk_size: int = 2**20,
    progress_bar: bool = True,
):
    """Get a file from a local folder, URL, or object store.

    Args:
        path (str): The path to the file to retreive.

            *   If ``object_store`` is specified, then the ``path`` should be the object name for the file to get.
                Do not include the the cloud provider or bucket name.

            *   If ``object_store`` is not specified but the ``path`` begins with ``http://`` or ``https://``,
                the object at this URL will be downloaded.

            *   Otherwise, ``path`` is presumed to be a local filepath.

        destination (str): The destination filepath.

            If ``path`` is a local filepath, then a symlink to ``path`` at ``destination`` will be created.
            Otherwise, ``path`` will be downloaded to a file at ``destination``.

        object_store (ObjectStore, optional): An :class:`~.ObjectStore`, if ``path`` is located inside
            an object store (i.e. AWS S3 or Google Cloud Storage). (default: ``None``)

            This :class:`~.ObjectStore` instance will be used to retreive the file. The ``path`` parameter
            should be set to the object name within the object store.

            Set this parameter to ``None`` (the default) if ``path`` is a URL or a local file.

        chunk_size (int, optional): Chunk size (in bytes). Ignored if ``path`` is a local file. (default: 1MB)

        progress_bar (bool, optional): Whether to show a progress bar. Ignored if ``path`` is a local file.
            (default: ``True``)

    Raises:
        GetFileNotFoundException: If the ``path`` does not exist, a ``GetFileNotFoundException`` exception will
            be raised.
    """
    if object_store is not None:
        if isinstance(object_store, ObjectStore):
            # Type ObjectStore
            try:
                total_size_in_bytes = object_store.get_object_size(path)
            except Exception as e:
                if "ObjectDoesNotExistError" in str(e):
                    raise GetFileNotFoundException(
                        f"Object name {path} not found in object store {object_store}") from e
                raise
            _write_to_file_with_pbar(
                destination=destination,
                total_size=total_size_in_bytes,
                iterator=object_store.download_object_as_stream(path, chunk_size=chunk_size),
                progress_bar=progress_bar,
                description=f"Downloading {path}",
            )
        else:
            # Type LoggerDestination
            object_store.get_file_artifact(artifact_name=path,
                                           destination=destination,
                                           chunk_size=chunk_size,
                                           progress_bar=progress_bar)
        return

    if path.lower().startswith("http://") or path.lower().startswith("https://"):
        # it's a url
        with requests.get(path, stream=True) as r:
            try:
                r.raise_for_status()
            except requests.exceptions.HTTPError as e:
                if r.status_code == 404:
                    raise GetFileNotFoundException(f"URL {path} not found") from e
                raise e
            total_size_in_bytes = r.headers.get('content-length')
            if total_size_in_bytes is not None:
                total_size_in_bytes = int(total_size_in_bytes)
            _write_to_file_with_pbar(
                destination,
                total_size=total_size_in_bytes,
                iterator=r.iter_content(chunk_size),
                progress_bar=progress_bar,
                description=f"Downloading {path}",
            )
        return

    # It's a local filepath
    if not os.path.exists(path):
        raise GetFileNotFoundException(f"Local path {path} does not exist")
    os.symlink(os.path.abspath(path), destination)


def _write_to_file_with_pbar(
    destination: str,
    total_size: Optional[int],
    iterator: Iterator[bytes],
    progress_bar: bool,
    description: str,
):
    """Write the contents of ``iterator`` to ``destination`` while showing a progress bar."""
    if progress_bar:
        if len(description) > 60:
            description = description[:42] + "..." + description[-15:]
        pbar = tqdm.tqdm(desc=description, total=total_size, unit='iB', unit_scale=True)
    else:
        pbar = None
    with open(destination, "wb") as fp:
        for chunk in iterate_with_pbar(iterator, pbar):
            fp.write(chunk)
