# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Abstract class for utilities that upload to and download from object stores."""

import abc
from typing import Iterator, Optional, Union

__all__ = ["ObjectStore", "ObjectStoreTransientError"]


class ObjectStoreTransientError(RuntimeError):
    """Custom exception class to signify transient errors.

    Implementations of the :class:`.ObjectStore` should re-raise any transient exceptions
    (e.g. too many requests, temporarily unavailable) with this class, so callers can easily
    detect whether they should attempt to retry any operation.

    For example, the :class:`.S3ObjectStore` does the following:

    .. testcode::

        from composer.utils import ObjectStore, ObjectStoreTransientError
        import botocore.exceptions

        class S3ObjectStore(ObjectStore):

            def upload_object(self, file_path: str, object_name: str):
                try:
                    ...
                except botocore.exceptions.ClientError as e:
                    if e.response['Error']['Code'] == 'LimitExceededException':
                        raise ObjectStoreTransientError(e.response['Error']['Code']) from e
                    raise e

    Then, callers can automatically handle exceptions:

    .. testcode::

        import time
        from composer.utils import ObjectStore, ObjectStoreTransientError

        def upload_file(object_store: ObjectStore, max_num_attempts: int = 3):
            for i in range(max_num_attempts):
                try:
                    object_store.upload_object(...)
                except ObjectStoreTransientError:
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


class ObjectStore(abc.ABC):
    """Abstract class for implementing object stores, such as LibcloudObjectStore and S3ObjectStore."""

    @abc.abstractmethod
    def get_uri(self, object_name: str) -> str:
        """Returns the URI for ``object_name``."""
        raise NotImplementedError(f"{type(self).__name__}.get_uri is not implemented")

    def upload_object(self, file_path: str, object_name: str):
        """Upload an object currently located on a disk.

        Args:
            file_path (str): Path the the object on disk
            object_name (str): Object name (where object will be stored in the container)
        """
        raise NotImplementedError(f"{type(self).__name__}.upload_object is not implemented")

    def upload_object_via_stream(self, obj: Union[bytes, Iterator[bytes]], object_name: str):
        """Upload an object.

        Args:
            obj (bytes | Iterator[bytes]): The object
            object_name (str): Object name (i.e. where the object will be stored in the container)
        """
        raise NotImplementedError(f"{type(self).__name__}.upload_object_via_stream is not implemented")

    def get_object_size(self, object_name: str) -> int:
        """Get the size of an object, in bytes.

        Args:
            object_name (str): The name of the object.

        Returns:
            int: The object size, in bytes.

        Raises:
            FileNotFoundError: If the file was not found in the object store.
            ObjectStoreTransientError: If there was a transient connection issue with getting the object size.
        """
        raise NotImplementedError(f"{type(self).__name__}.get_object_size is not implemented")

    def download_object(
        self,
        object_name: str,
        destination_path: str,
        chunk_size: Optional[int] = None,
        overwrite_existing: bool = False,
    ):
        """Download an object to the specified destination path.

        Args:
            object_name (str): The name of the object to download.
            destination_path (str): Full path to a file or a directory where the incoming file will be saved.
            chunk_size (Optional[int], optional): Optional chunk size (in bytes).
            overwrite_existing (bool, optional): Whether to overwrite an existing file at ``destination_path``, if it exists.
                (default: ``False``)

        Raises:
            FileNotFoundError: If the file was not found in the object store.
            ObjectStoreTransientError: If there was a transient connection issue with downloading the object.
        """
        raise NotImplementedError(f"{type(self).__name__}.download_object is not implemented")

    def download_object_as_stream(self, object_name: str, chunk_size: Optional[int] = None):
        """Return a iterator which yields object data.

        Args:
            object_name (str): Object name.
            chunk_size (Optional[int], optional): Optional chunk size (in bytes).

        Returns:
            Iterator[bytes]: The object, as a byte stream.

        Raises:
            FileNotFoundError: If the file was not found in the object store.
            ObjectStoreTransientError: If there was a transient connection issue with downloading the object.
        """
        raise NotImplementedError(f"{type(self).__name__}.download_object_as_stream is not implemented")
