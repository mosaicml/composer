# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Helpers for working with files."""

from __future__ import annotations

import os
import pathlib
import re
import tempfile
import uuid
from typing import TYPE_CHECKING, Optional, Union

import requests
import tqdm

from composer.core.time import Time, Timestamp
from composer.utils import dist
from composer.utils.iter_helpers import iterate_with_callback
from composer.utils.object_store import ObjectStore

if TYPE_CHECKING:
    from composer.loggers import LoggerDestination

__all__ = [
    'get_file',
    'ensure_folder_is_empty',
    'ensure_folder_has_no_conflicting_files',
    'format_name_with_dist',
    'format_name_with_dist_and_time',
    'is_tar',
    'create_symlink_file',
]


def is_tar(name: Union[str, pathlib.Path]) -> bool:
    """Returns whether ``name`` has a tar-like extension.

    Args:
        name (str | pathlib.Path): The name to check.

    Returns:
        bool: Whether ``name`` is a tarball.
    """
    return any(str(name).endswith(x) for x in ('.tar', '.tgz', '.tar.gz', '.tar.bz2', '.tar.lzma'))


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
            if not file.startswith('.'):
                raise FileExistsError(f'{folder_name} is not empty; {os.path.join(root, file)} exists.')


def ensure_folder_has_no_conflicting_files(folder_name: Union[str, pathlib.Path], filename: str, timestamp: Timestamp):
    """Ensure that the given folder does not have any files conflicting with the ``filename`` format string.

    If any filename is formatted with a timestamp where the epoch, batch, sample, or token counts are after
    ``timestamp``, a ``FileExistsError`` will be raised.
    If ``filename`` and occurs later than ``timestamp``, raise a ``FileExistsError``.

    Args:
        folder_name (str | pathlib.Path): The folder to inspect.
        filename (str): The pattern string for potential files.
        timestamp (Timestamp): Ignore any files that occur before the provided timestamp.

    Raises:
        FileExistsError: If ``folder_name`` contains any files matching the ``filename`` template before ``timestamp``.
    """
    # Prepare regex pattern by replacing f-string formatting with regex.
    pattern = f'^{filename}$'
    # Format time vars for capture
    time_names = ['epoch', 'batch', 'sample', 'token', 'batch_in_epoch', 'sample_in_epoch', 'token_in_epoch']
    captured_names = {time_name: f'{{{time_name}}}' in filename for time_name in time_names}
    for time_name, is_captured in captured_names.items():
        if is_captured:
            pattern = pattern.replace(f'{{{time_name}}}', f'(?P<{time_name}>\\d+)')
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
            if captured_names['token'] and Time.from_token(int(match.group('token'))) > timestamp.token:
                valid_match = False
            elif captured_names['sample'] and Time.from_sample(int(match.group('sample'))) > timestamp.sample:
                valid_match = False
            elif captured_names['batch'] and Time.from_batch(int(match.group('batch'))) > timestamp.batch:
                valid_match = False
            elif captured_names['epoch'] and Time.from_epoch(int(match.group('epoch'))) > timestamp.epoch:
                valid_match = False
            # If epoch count is same, check batch_in_epoch, sample_in_epoch, token_in_epoch
            elif captured_names['epoch'] and Time.from_epoch(int(match.group('epoch'))) == timestamp.epoch:
                if captured_names['token_in_epoch'] and Time.from_token(int(
                        match.group('token_in_epoch'))) > timestamp.token_in_epoch:
                    valid_match = False
                elif captured_names['sample_in_epoch'] and Time.from_sample(int(
                        match.group('sample_in_epoch'))) > timestamp.sample_in_epoch:
                    valid_match = False
                elif captured_names['batch_in_epoch'] and Time.from_batch(int(
                        match.group('batch_in_epoch'))) > timestamp.batch_in_epoch:
                    valid_match = False
            if not valid_match:
                raise FileExistsError(
                    f'{os.path.join(folder_name, file)} exists and conflicts in namespace with a future checkpoint of the current run.'
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


def format_name_with_dist(format_str: str, run_name: str, **extra_format_kwargs: object):  # noqa: D103
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
+----------------------------+------------------------------------------------------------+
| Variable                   | Description                                                |
+============================+============================================================+
| ``{run_name}``             | The name of the training run. See                          |
|                            | :attr:`~composer.loggers.logger.Logger.run_name`.          |
+----------------------------+------------------------------------------------------------+
| ``{rank}``                 | The global rank, as returned by                            |
|                            | :func:`~composer.utils.dist.get_global_rank`.              |
+----------------------------+------------------------------------------------------------+
| ``{local_rank}``           | The local rank of the process, as returned by              |
|                            | :func:`~composer.utils.dist.get_local_rank`.               |
+----------------------------+------------------------------------------------------------+
| ``{world_size}``           | The world size, as returned by                             |
|                            | :func:`~composer.utils.dist.get_world_size`.               |
+----------------------------+------------------------------------------------------------+
| ``{local_world_size}``     | The local world size, as returned by                       |
|                            | :func:`~composer.utils.dist.get_local_world_size`.         |
+----------------------------+------------------------------------------------------------+
| ``{node_rank}``            | The node rank, as returned by                              |
|                            | :func:`~composer.utils.dist.get_node_rank`.                |
+----------------------------+------------------------------------------------------------+
| ``{epoch}``                | The total epoch count, as returned by                      |
|                            | :meth:`~composer.core.time.Timestamp.epoch`.               |
+----------------------------+------------------------------------------------------------+
| ``{batch}``                | The total batch count, as returned by                      |
|                            | :meth:`~composer.core.time.Timestamp.batch`.               |
+----------------------------+------------------------------------------------------------+
| ``{batch_in_epoch}``       | The batch count in the current epoch, as returned by       |
|                            | :meth:`~composer.core.time.Timestamp.batch_in_epoch`.      |
+----------------------------+------------------------------------------------------------+
| ``{sample}``               | The total sample count, as returned by                     |
|                            | :meth:`~composer.core.time.Timestamp.sample`.              |
+----------------------------+------------------------------------------------------------+
| ``{sample_in_epoch}``      | The sample count in the current epoch, as returned by      |
|                            | :meth:`~composer.core.time.Timestamp.sample_in_epoch`.     |
+----------------------------+------------------------------------------------------------+
| ``{token}``                | The total token count, as returned by                      |
|                            | :meth:`~composer.core.time.Timestamp.token`.               |
+----------------------------+------------------------------------------------------------+
| ``{token_in_epoch}``       | The token count in the current epoch, as returned by       |
|                            | :meth:`~composer.core.time.Timestamp.token_in_epoch`.      |
+----------------------------+------------------------------------------------------------+
| ``{total_wct}``            | The total training duration in seconds, as returned by     |
|                            | :meth:`~composer.core.time.Timestamp.total_wct`.           |
+----------------------------+------------------------------------------------------------+
| ``{epoch_wct}``            | The epoch duration in seconds, as returned by              |
|                            | :meth:`~composer.core.time.Timestamp.epoch_wct`.           |
+----------------------------+------------------------------------------------------------+
| ``{batch_wct}``            | The batch duration in seconds, as returned by              |
|                            | :meth:`~composer.core.time.Timestamp.batch_wct`.           |
+----------------------------+------------------------------------------------------------+
"""


def format_name_with_dist_and_time(
    format_str: str,
    run_name: str,
    timestamp: Timestamp,
    **extra_format_kwargs: object,
):  # noqa: D103
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
        total_wct=timestamp.total_wct.total_seconds(),
        epoch_wct=timestamp.epoch_wct.total_seconds(),
        batch_wct=timestamp.batch_wct.total_seconds(),
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
    overwrite: bool = False,
    progress_bar: bool = True,
):
    """Get a file from a local folder, URL, or object store.

    Args:
        path (str): The path to the file to retrieve.

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

            This :class:`~.ObjectStore` instance will be used to retrieve the file. The ``path`` parameter
            should be set to the object name within the object store.

            Set this parameter to ``None`` (the default) if ``path`` is a URL or a local file.

        overwrite (bool): Whether to overwrite an existing file at ``destination``. (default: ``False``)

        progress_bar (bool, optional): Whether to show a progress bar. Ignored if ``path`` is a local file.
            (default: ``True``)

    Raises:
        FileNotFoundError: If the ``path`` does not exist.
    """
    if path.endswith('.symlink'):
        with tempfile.TemporaryDirectory() as tmpdir:
            symlink_file_name = os.path.join(tmpdir, 'file.symlink')
            # Retrieve the symlink
            _get_file(
                path=path,
                destination=symlink_file_name,
                object_store=object_store,
                overwrite=False,
                progress_bar=progress_bar,
            )
            # Read object name in the symlink
            with open(symlink_file_name, 'r') as f:
                real_path = f.read()

        # Recurse
        return get_file(
            path=real_path,
            destination=destination,
            object_store=object_store,
            overwrite=overwrite,
            progress_bar=progress_bar,
        )

    try:
        _get_file(
            path=path,
            destination=destination,
            object_store=object_store,
            overwrite=overwrite,
            progress_bar=progress_bar,
        )
    except FileNotFoundError as e:
        new_path = path + '.symlink'
        try:
            # Follow the symlink
            return get_file(
                path=new_path,
                destination=destination,
                object_store=object_store,
                overwrite=overwrite,
                progress_bar=progress_bar,
            )
        except FileNotFoundError as ee:
            # Raise the original not found error first, which contains the path to the user-specified file
            raise e from ee


def _get_file(
    path: str,
    destination: str,
    object_store: Optional[Union[ObjectStore, LoggerDestination]],
    overwrite: bool,
    progress_bar: bool,
):
    # Underlying _get_file logic that does not deal with symlinks
    if object_store is not None:
        if isinstance(object_store, ObjectStore):
            total_size_in_bytes = object_store.get_object_size(path)
            object_store.download_object(
                object_name=path,
                filename=destination,
                callback=_get_callback(f'Downloading {path}') if progress_bar else None,
                overwrite=overwrite,
            )
        else:
            # Type LoggerDestination
            object_store.get_file_artifact(
                artifact_name=path,
                destination=destination,
                progress_bar=progress_bar,
                overwrite=overwrite,
            )
        return

    if path.lower().startswith('http://') or path.lower().startswith('https://'):
        # it's a url
        with requests.get(path, stream=True) as r:
            try:
                r.raise_for_status()
            except requests.exceptions.HTTPError as e:
                if r.status_code == 404:
                    raise FileNotFoundError(f'URL {path} not found') from e
                raise e
            total_size_in_bytes = r.headers.get('content-length')
            if total_size_in_bytes is not None:
                total_size_in_bytes = int(total_size_in_bytes)
            else:
                total_size_in_bytes = 0

            tmp_path = destination + f'.{uuid.uuid4()}.tmp'
            try:
                with open(tmp_path, 'wb') as f:
                    for data in iterate_with_callback(
                            r.iter_content(2**20),
                            total_size_in_bytes,
                            callback=_get_callback(f'Downloading {path}') if progress_bar else None,
                    ):
                        f.write(data)
            except:
                # The download failed for some reason. Make a best-effort attempt to remove the temporary file.
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
                raise
            else:
                os.rename(tmp_path, destination)
        return

    # It's a local filepath
    if not os.path.exists(path):
        raise FileNotFoundError(f'Local path {path} does not exist')
    os.symlink(os.path.abspath(path), destination)


def _get_callback(description: str):
    if len(description) > 60:
        description = description[:42] + '...' + description[-15:]
    pbar = None

    def callback(num_bytes: int, total_size: int):
        nonlocal pbar
        if num_bytes == 0 or pbar is None:
            pbar = tqdm.tqdm(desc=description, total=total_size, unit='iB', unit_scale=True)
        n = num_bytes - pbar.n
        pbar.update(n)
        if num_bytes == total_size:
            pbar.close()

    return callback


def create_symlink_file(
    existing_path: str,
    destination_filename: Union[str, pathlib.Path],
):
    """Create a symlink file, which can be followed by :func:`get_file`.

    Unlike unix symlinks, symlink files can be created by this function are normal text files and can be
    uploaded to object stores via :meth:`.ObjectStore.upload_object` or loggers via :meth:`.Logger.file_artifact`
    that otherwise would not support unix-style symlinks.

    Args:
        existing_path (str): The name of existing object that the symlink file should point to.
        destination_filename (str | pathlib.Path): The filename to which to write the symlink.
            It must end in ``'.symlink'``.
    """
    # Loggers might not natively support symlinks, so we emulate symlinks via text files ending with `.symlink`
    # This text file contains the name of the object it is pointing to.
    # Only symlink if we're logging artifact to begin with
    # Write artifact name into file to emulate symlink
    # Add .symlink extension so we can identify as emulated symlink when downloading
    destination_filename = str(destination_filename)
    if not destination_filename.endswith('.symlink'):
        raise ValueError('The symlink filename must end with .symlink.')
    with open(destination_filename, 'x') as f:
        f.write(existing_path)
