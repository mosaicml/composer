# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Abstract class for utilities that upload to and download from remote filesystems."""

import abc
import pathlib
from types import TracebackType
from typing import Callable, Optional, Type, Union

__all__ = ['RemoteFilesystem', 'RemoteFilesystemTransientError']


class RemoteFilesystemTransientError(RuntimeError):
    """Custom exception class to signify transient errors.

    Implementations of the :class:`.RemoteFilesystem` should re-raise any transient exceptions
    (e.g. too many requests, temporarily unavailable) with this class, so callers can easily
    detect whether they should attempt to retry any operation.

    For example, the :class:`.S3RemoteFilesystem` does the following:

    .. testcode::

        from composer.utils import RemoteFilesystem, RemoteFilesystemTransientError
        import botocore.exceptions

        class S3RemoteFilesystem(RemoteFilesystem):

            def upload_file(self, file_path: str, remote_file_name: str):
                try:
                    ...
                except botocore.exceptions.ClientError as e:
                    if e.response['Error']['Code'] == 'LimitExceededException':
                        raise RemoteFilesystemTransientError(e.response['Error']['Code']) from e
                    raise e

    Then, callers can automatically handle exceptions:

    .. testcode::

        import time
        from composer.utils import RemoteFilesystem, RemoteFilesystemTransientError

        def upload_file(remote_filesystem: RemoteFilesystem, max_num_attempts: int = 3):
            for i in range(max_num_attempts):
                try:
                    remote_filesystem.upload_file(...)
                except RemoteFilesystemTransientError:
                    if i + 1 == max_num_attempts:
                        raise
                    else:
                        # Try again after exponential back-off
                        time.sleep(2**i)
                else:
                    # upload successful
                    return
    """
    pass


class RemoteFilesystem(abc.ABC):
    """Abstract class for implementing remote filesystem backends, such as LibcloudRemoteFilesystem and S3RemoteFilesystem."""

    def get_uri(self, remote_file_name: str) -> str:
        """Returns the URI for ``remote_file_name``.

        .. note::

            This function does not check that ``remote_file_name`` is in the remote filesystem.
            It computes the URI statically.

        Args:
            remote_file_name (str): The object name.

        Returns:
            str: The URI for ``remote_file_name`` in the remote filesystem.
        """
        raise NotImplementedError(f'{type(self).__name__}.get_uri is not implemented')

    def upload_file(
        self,
        remote_file_name: str,
        filename: Union[str, pathlib.Path],
        callback: Optional[Callable[[int, int], None]] = None,
    ) -> None:
        """Upload an object currently located on a disk.

        Args:
            remote_file_name (str): Object name (where object will be stored in the container)
            filename (str | pathlib.Path): Path the the object on disk
            callback ((int, int) -> None, optional): If specified, the callback is periodically called with the number of bytes
                uploaded and the total size of the object being uploaded.

        Raises:
            RemoteFilesystemTransientError: If there was a transient connection issue with uploading the object.
        """
        del remote_file_name, filename, callback  # unused
        raise NotImplementedError(f'{type(self).__name__}.upload_file is not implemented')

    def get_file_size(self, remote_file_name: str) -> int:
        """Get the size of an object, in bytes.

        Args:
            remote_file_name (str): The name of the object.

        Returns:
            int: The object size, in bytes.

        Raises:
            FileNotFoundError: If the file was not found in the remote filesystem.
            RemoteFilesystemTransientError: If there was a transient connection issue with getting the object size.
        """
        raise NotImplementedError(f'{type(self).__name__}.get_file_size is not implemented')

    def download_file(
        self,
        remote_file_name: str,
        filename: Union[str, pathlib.Path],
        overwrite: bool = False,
        callback: Optional[Callable[[int, int], None]] = None,
    ) -> None:
        """Download an object to the specified destination path.

        Args:
            remote_file_name (str): The name of the object to download.
            filename (str | pathlib.Path): Full path to a file or a directory where the incoming file will be saved.
            overwrite (bool, optional): Whether to overwrite an existing file at ``filename``, if it exists.
                (default: ``False``)
            callback ((int) -> None, optional): If specified, the callback is periodically called with the number of bytes already
                downloaded and the total size of the object.

        Raises:
            FileNotFoundError: If the file was not found in the remote filesystem.
            RemoteFilesystemTransientError: If there was a transient connection issue with downloading the object.
        """
        del remote_file_name, filename, overwrite, callback  # unused
        raise NotImplementedError(f'{type(self).__name__}.download_file is not implemented')

    def close(self):
        """Close the remote filesystem."""
        pass

    def __enter__(self):
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc: Optional[BaseException],
        traceback: Optional[TracebackType],
    ):
        del exc_type, exc, traceback  # unused
        self.close()
