# Copyright 2021 MosaicML. All Rights Reserved.

"""Utility for retrieving files."""

import os
from typing import Iterator, Optional

import requests
import tqdm

from composer.utils.iter_helpers import iterate_with_pbar
from composer.utils.object_store import ObjectStoreProvider

__all__ = ['GetFileNotFoundException', 'get_file']


class GetFileNotFoundException(RuntimeError):
    """Exception if :meth:`get_file` failed due to a not found error."""
    pass


def get_file(
    path: str,
    destination: str,
    object_store: Optional[ObjectStoreProvider] = None,
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

        object_store (ObjectStoreProvider, optional): An :class:`~.ObjectStoreProvider`, if ``path`` is located inside
            an object store (i.e. AWS S3 or Google Cloud Storage). (default: ``None``)

            This :class:`~.ObjectStoreProvider` instance will be used to retreive the file. The ``path`` parameter
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
        try:
            total_size_in_bytes = object_store.get_object_size(path)
        except Exception as e:
            if "ObjectDoesNotExistError" in str(e):
                raise GetFileNotFoundException(f"Object name {path} not found in object store {object_store}") from e
            raise
        _write_to_file_with_pbar(
            destination=destination,
            total_size=total_size_in_bytes,
            iterator=object_store.download_object_as_stream(path, chunk_size=chunk_size),
            progress_bar=progress_bar,
            description=f"Downloading {path}",
        )
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
