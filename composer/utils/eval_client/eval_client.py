# Copyright 2023 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Abstract class for utilities that access and run code on serverless eval clients."""

import abc
import pathlib
from types import TracebackType
from typing import Callable, Optional, Type, Union

__all__ = ['EvalClient']


class EvalClient(abc.ABC):
    """Abstract class for implementing eval clients, such as LambdaEvalClient."""

    def get_uri(self, object_name: str) -> str:
        """Returns the URI for ``object_name``.

        .. note::

            This function does not check that ``object_name`` is in the object store.
            It computes the URI statically.

        Args:
            object_name (str): The object name.

        Returns:
            str: The URI for ``object_name`` in the object store.
        """
        raise NotImplementedError(f'{type(self).__name__}.get_uri is not implemented')


    def download_object(
        self,
        object_name: str,
        filename: Union[str, pathlib.Path],
        overwrite: bool = False,
        callback: Optional[Callable[[int, int], None]] = None,
    ) -> None:
        """Download an object to the specified destination path.

        Args:
            object_name (str): The name of the object to download.
            filename (str | pathlib.Path): Full path to a file or a directory where the incoming file will be saved.
            overwrite (bool, optional): Whether to overwrite an existing file at ``filename``, if it exists.
                (default: ``False``)
            callback ((int) -> None, optional): If specified, the callback is periodically called with the number of bytes already
                downloaded and the total size of the object.

        Raises:
            FileNotFoundError: If the file was not found in the object store.
            ObjectStoreTransientError: If there was a transient connection issue with downloading the object.
        """
        del object_name, filename, overwrite, callback  # unused
        raise NotImplementedError(f'{type(self).__name__}.download_object is not implemented')

    def close(self):
        """Close the object store."""
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