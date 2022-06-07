# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Abstract class for utilities that upload to and download from object stores."""

import abc
from typing import Iterator, Optional, Union

__all__ = ["ObjectStore"]


class ObjectStore(abc.ABC):
    """Abstract class for implementing object stores, such as LibcloudObjectStore and S3ObjectStore."""

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
        """
        raise NotImplementedError(f"{type(self).__name__}.download_object is not implemented")

    def download_object_as_stream(self, object_name: str, chunk_size: Optional[int] = None):
        """Return a iterator which yields object data.

        Args:
            object_name (str): Object name.
            chunk_size (Optional[int], optional): Optional chunk size (in bytes).

        Returns:
            Iterator[bytes]: The object, as a byte stream.
        """
        raise NotImplementedError(f"{type(self).__name__}.download_object_as_stream is not implemented")
