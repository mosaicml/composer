# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Helpers for working with files."""

from __future__ import annotations

import logging
import os
import pathlib
import re
import tempfile
import uuid
import warnings
from typing import TYPE_CHECKING, Any, Callable, Optional, Union
from urllib.parse import urlparse

import requests
import tqdm

from composer.utils import dist
from composer.utils.iter_helpers import iterate_with_callback
from composer.utils.misc import partial_format
from composer.utils.object_store import (
    GCSObjectStore,
    LibcloudObjectStore,
    MLFlowObjectStore,
    ObjectStore,
    OCIObjectStore,
    S3ObjectStore,
    UCObjectStore,
)
from composer.utils.object_store.mlflow_object_store import MLFLOW_DBFS_PATH_PREFIX

if TYPE_CHECKING:
    from composer.core import Timestamp
    from composer.loggers import LoggerDestination, RemoteUploaderDownloader

log = logging.getLogger(__name__)

__all__ = [
    'get_file',
    'ensure_folder_is_empty',
    'ensure_folder_has_no_conflicting_files',
    'format_name_with_dist',
    'format_name_with_dist_and_time',
    'is_tar',
    'create_symlink_file',
    'maybe_create_object_store_from_uri',
    'maybe_create_remote_uploader_downloader_from_uri',
    'parse_uri',
    'extract_path_from_symlink',
    'validate_credentials',
]


def extract_path_from_symlink(
    source_path: str,
    object_store: Optional[Union[LoggerDestination, ObjectStore]] = None,
) -> str:
    """Returns the checkpont path from symlink file.

    Args:
        source_path(str): The remote symlink path.
        object_store(LoggerDestination | ObjectStore, optional): The object store
            used to download the remote symlink file

    Returns:
        str: The content of the remote symlink file.
    """
    if object_store is not None:
        with tempfile.TemporaryDirectory() as tmpdir:
            _, _, source_path = parse_uri(source_path)
            symlink_file_path = os.path.join(tmpdir, 'file.symlink')
            if isinstance(object_store, ObjectStore):
                object_store.download_object(object_name=source_path, filename=symlink_file_path)
            elif isinstance(object_store, LoggerDestination):
                object_store.download_file(remote_file_name=source_path, destination=symlink_file_path)
            with open(symlink_file_path, 'r') as f:
                real_path = f.read()
                log.debug(f'Read path {real_path} from symlink file.')
    else:
        real_path = os.readlink(source_path)

    return real_path


def _get_dist_config(strict: bool = True) -> dict[str, Any]:
    """Returns a dict of distributed settings (rank, world_size, etc.).

    If ``strict=True``, will error if a setting is not available (e.g. the
    environment variable is not set). Otherwise, will only return settings
    that are available.
    """
    settings = {
        'rank': dist.get_global_rank,
        'local_rank': dist.get_local_rank,
        'world_size': dist.get_world_size,
        'local_world_size': dist.get_local_world_size,
        'node_rank': dist.get_node_rank,
    }

    dist_config = {}
    for name, func in settings.items():
        try:
            value = func()
        except dist.MissingEnvironmentError as e:
            if strict:
                raise e
        else:
            dist_config[name] = value

    return dist_config


def is_tar(name: Union[str, pathlib.Path]) -> bool:
    """Returns whether ``name`` has a tar-like extension.

    Args:
        name (str | pathlib.Path): The name to check.

    Returns:
        bool: Whether ``name`` is a tarball.
    """
    parts = str(name).split('.')
    return len(parts) > 1 and ('tar' in parts[-2:] or parts[-1] == 'tgz')


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

    # Format time vars for regex match
    for unit in ['epoch', 'batch', 'sample', 'token', 'batch_in_epoch', 'sample_in_epoch', 'token_in_epoch']:
        if unit in filename:
            pattern = pattern.replace(f'{{{unit}}}', f'(?P<{unit}>\\d+)')

    # Format rank information
    pattern = pattern.format(**_get_dist_config(strict=False))

    template = re.compile(pattern)

    for file in os.listdir(folder_name):
        match = template.match(file)

        if match is not None:
            match = match.groupdict()
            for unit, value in match.items():
                if unit.endswith('_in_epoch'):
                    if 'epoch' not in match:
                        raise ValueError(f'{filename} has {{unit}} but not {{epoch}}. Add {{epoch}} for uniqueness.')
                    if int(match['epoch']) != timestamp.epoch:
                        continue  # only check _in_epoch if both files have same epoch count

                if int(value) > int(getattr(timestamp, unit)):
                    raise FileExistsError(
                        f'{os.path.join(folder_name, file)} may conflict with a future checkpoint of the current run.'
                        'Please delete that file, change to a new folder, or set overwrite=True.',
                    )


FORMAT_NAME_WITH_DIST_TABLE = """
+------------------------+-------------------------------------------------------+
| Variable               | Description                                           |
+========================+=======================================================+
| ``{run_name}``         | The name of the training run. See                     |
|                        | :attr:`.Logger.run_name`.                             |
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
    formatted_str = partial_format(
        format_str,
        run_name=run_name,
        **_get_dist_config(strict=False),
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
|                            | :attr:`.Logger.run_name`.                                  |
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
    formatted_str = partial_format(
        format_str,
        run_name=run_name,
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
        **_get_dist_config(strict=False),
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


def is_uri(path: str) -> bool:
    """Check if the path is a URI.

    Args:
        path (str): The path to check.

    Returns:
        bool: Whether the path is a URI.
    """
    backend, _, _ = parse_uri(path)
    return backend != ''


def parse_uri(uri: str) -> tuple[str, str, str]:
    """Uses :py:func:`urllib.parse.urlparse` to parse the provided URI.

    Args:
        uri (str): The provided URI string

    Returns:
        tuple[str, str, str]: A tuple containing the backend (e.g. s3), bucket name, and path.
                              Backend name will be empty string if the input is a local path
    """
    if isinstance(uri, pathlib.Path):
        uri = str(uri)
    uri = uri.replace('AZURE_BLOBS', 'azure')  # urlparse does not support _ in scheme
    parse_result = urlparse(uri)
    backend, net_loc, path = parse_result.scheme, parse_result.netloc, parse_result.path
    bucket_name = net_loc if '@' not in net_loc else net_loc.split('@')[0]
    if backend == '':
        return backend, bucket_name, path
    else:
        return backend, bucket_name, path.lstrip('/')


# Dictionary mapping backend names to ObjectStore factory functions
BACKEND_TO_OBJECT_STORE_FACTORY: dict[str, Callable[[str, str], ObjectStore]] = {
    's3':
        lambda bucket, path: S3ObjectStore(bucket=bucket),
    'gs':
        lambda bucket, path: GCSObjectStore(bucket=bucket),
    'oci':
        lambda bucket, path: OCIObjectStore(bucket=bucket),
    'azure':
        lambda bucket, path: LibcloudObjectStore(
            provider='AZURE_BLOBS',
            container=bucket,
            key_environ='AZURE_ACCOUNT_NAME',
            secret_environ='AZURE_ACCOUNT_ACCESS_KEY',
        ),
}


def maybe_create_object_store_from_uri(uri: str) -> Optional[ObjectStore]:
    """Automatically creates an :class:`composer.utils.ObjectStore` from supported URI formats.

    Currently supported backends are ``s3://``, ``oci://``, and local paths (in which case ``None`` will be returned)

    Args:
        uri (str): The path to (maybe) create an :class:`composer.utils.ObjectStore` from.

    Raises:
        NotImplementedError: Raises when the URI format is not supported.

    Returns:
        Optional[ObjectStore]: Returns an :class:`composer.utils.ObjectStore` if the URI is of a supported format, otherwise None
    """
    backend, bucket_name, path = parse_uri(uri)

    # If backend is empty, assume local path and return None
    if backend == '':
        return None

    # Handle special cases like WandB, MLFlow, etc.
    elif backend == 'wandb':
        raise NotImplementedError(
            f'There is no implementation for WandB load_object_store via URI. Please use WandBLogger',
        )
    elif backend == 'dbfs':
        if path.startswith(MLFLOW_DBFS_PATH_PREFIX):
            store = None
            if dist.get_global_rank() == 0:
                store = MLFlowObjectStore(path)

                # The path may have had placeholders, so update it with the experiment/run IDs initialized by the store
                path = store.get_dbfs_path(path)

            # Broadcast the rank 0 updated path to all ranks for their own object stores
            path_list = [path]
            dist.broadcast_object_list(path_list, src=0)
            path = path_list[0]

            # Create the object store for all other ranks
            if dist.get_global_rank() != 0:
                store = MLFlowObjectStore(path)
            return store
        else:
            # validate if the path conforms to the requirements for UC volume paths
            UCObjectStore.validate_path(path)
            return UCObjectStore(path=path)

    # Check if backend is registered
    elif backend in BACKEND_TO_OBJECT_STORE_FACTORY:
        return BACKEND_TO_OBJECT_STORE_FACTORY[backend](bucket_name, path)

    # If backend is unknown, raise NotImplementedError
    raise NotImplementedError(
        f'There is no implementation for the cloud backend {backend} via URI. Please use '
        'one of the supported object stores',
    )


def maybe_create_remote_uploader_downloader_from_uri(
    uri: str,
    loggers: list[LoggerDestination],
) -> Optional['RemoteUploaderDownloader']:
    """Automatically creates a :class:`composer.loggers.RemoteUploaderDownloader` from supported URI formats.

    Currently supported backends are ``s3://``, ``oci://``, and local paths (in which case ``None`` will be returned)

    Args:
        uri (str):The path to (maybe) create a :class:`composer.loggers.RemoteUploaderDownloader` from
        loggers (list[:class:`composer.loggers.LoggerDestination`]): List of the existing :class:`composer.loggers.LoggerDestination` s so as to not create a duplicate

    Raises:
        NotImplementedError: Raises when the URI format is not supported.

    Returns:
        Optional[RemoteUploaderDownloader]: Returns a :class:`composer.loggers.RemoteUploaderDownloader` if the URI is of a supported format, otherwise None
    """
    from composer.loggers import RemoteUploaderDownloader
    existing_remote_uds = [logger_dest for logger_dest in loggers if isinstance(logger_dest, RemoteUploaderDownloader)]
    backend, bucket_name, path = parse_uri(uri)
    if backend == '':
        return None
    for existing_remote_ud in existing_remote_uds:
        if ((existing_remote_ud.remote_backend_name == backend) and
            (existing_remote_ud.remote_bucket_name == bucket_name)):
            warnings.warn(
                f'There already exists a RemoteUploaderDownloader object to handle the uri: {uri} you specified',
            )
            return None
    if backend in ['s3', 'oci', 'gs']:
        return RemoteUploaderDownloader(bucket_uri=f'{backend}://{bucket_name}')
    elif backend == 'azure':
        return RemoteUploaderDownloader(
            bucket_uri=f'libcloud://{bucket_name}',
            backend_kwargs={
                'provider': 'AZURE_BLOBS',
                'container': bucket_name,
                'key_environ': 'AZURE_ACCOUNT_NAME',
                'secret_environ': 'AZURE_ACCOUNT_ACCESS_KEY',
            },
        )
    elif backend == 'dbfs':
        return RemoteUploaderDownloader(bucket_uri=uri, backend_kwargs={'path': path})
    elif backend == 'wandb':
        raise NotImplementedError(
            f'There is no implementation for WandB via URI. Please use '
            'WandBLogger with log_artifacts set to True',
        )
    else:
        raise NotImplementedError(
            f'There is no implementation for the cloud backend {backend} via URI. Please use '
            'one of the supported RemoteUploaderDownloader object stores',
        )


def list_remote_objects(remote_path: str) -> list[str]:
    """List objects at the remote path.

    Args:
        remote_path (str): Remote object store path.

    Returns:
        A list of objects at the remote path.
    """
    object_store = maybe_create_object_store_from_uri(remote_path)
    if object_store is None:
        raise ValueError(f'Failed to create object store. The given path {remote_path} is a local path.')
    _, _, prefix = parse_uri(remote_path)
    objects = object_store.list_objects(prefix)
    return objects


def validate_remote_path():
    """Entry point to composer_validate_remote_path cli command.

    Validates a remote path.
    If the remote path is valid, prints a list of objects at the path.
    Otherwise, raises an error.
    """
    import sys
    args = sys.argv
    if len(args) == 1:
        raise ValueError('Please provide a remote path.')
    if len(args) > 2:
        raise ValueError('Extra arguments found. Please provide only one remote path.')
    remote_path = sys.argv[1]
    objects = list_remote_objects(remote_path)
    if len(objects) == 0:
        raise ValueError(f'No objects at path {remote_path} found. Please check your path and your access credentials.')
    objects_str = '\n'.join(objects)
    print(f'Found {len(objects)} objects at {remote_path} \n{objects_str}')


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

            *   If ``object_store`` is not specified, but the ``path`` begins with ``s3://``, or another backend
                supported by :meth:`composer.utils.maybe_create_object_store_from_uri` an appropriate object store
                will be created and used.

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
    if object_store is None and not (path.lower().startswith('http://') or path.lower().startswith('https://')):
        object_store = maybe_create_object_store_from_uri(path)
        _, _, path = parse_uri(path)

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
                log.debug(f'Read path {real_path} from symlink file.')

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
            object_store.download_file(
                remote_file_name=path,
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

    if os.path.exists(destination) and overwrite:
        os.remove(destination)

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
    uploaded to object stores via :meth:`.ObjectStore.upload_object` or loggers via :meth:`.Logger.upload_file`
    that otherwise would not support unix-style symlinks.

    Args:
        existing_path (str): The name of existing object that the symlink file should point to.
        destination_filename (str | pathlib.Path): The filename to which to write the symlink.
            It must end in ``'.symlink'``.
    """
    # Loggers might not natively support symlinks, so we emulate symlinks via text files ending with `.symlink`
    # This text file contains the name of the object it is pointing to.
    # Only symlink if we're uploading files to begin with
    # Write remote file name into file to emulate symlink
    # Add .symlink extension so we can identify as emulated symlink when downloading
    destination_filename = str(destination_filename)
    if not destination_filename.endswith('.symlink'):
        raise ValueError('The symlink filename must end with .symlink.')
    with open(destination_filename, 'x') as f:
        f.write(existing_path)


def validate_credentials(
    remote_backend: ObjectStore,
    remote_file_name_to_test: str,
):
    """Upload a tiny text file to test if the credentials are setup correctly."""
    # Validates the credentials by attempting to touch a file in the bucket
    # raises an error if there was a credentials failure.
    with tempfile.NamedTemporaryFile('wb') as f:
        f.write(b'credentials_validated_successfully')
        remote_backend.upload_object(
            object_name=remote_file_name_to_test,
            filename=f.name,
        )
