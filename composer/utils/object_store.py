# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Abstract class for utilities that upload to and download from object stores."""

from typing import Iterator, Union
import abc

__all__ = ["ObjectStore"]


class ObjectStore(abc.ABC):
    """Abstract class for implementing object stores, such as LibcloudObjectStore and S3ObjectStore"""

    def upload_object(self, file_path: str, object_name: str):
        """Upload an object currently located on a disk.

        Args:
            file_path (str): Path the the object on disk
            object_name (str): Object name (where object will be stored in the container)
        """
        pass

    def upload_object_via_stream(self, obj: Union[bytes, Iterator[bytes]], object_name: str):
        """Upload an object.

        Args:
            obj (bytes | Iterator[bytes]): The object
            object_name (str): Object name (i.e. where the object will be stored in the container)
        """
        pass

    def download_object(
        self,
        object_name: str,
        destination_path: str,
    ):
        """Download an object to the specified destination path.
        
        Args:
            object_name (str): The name of the object to download.
            destination_path (str): Full path to a file or a directory where the incoming file will be saved.
        """
        pass

    def download_object_as_stream(self, object_name: str):
        """Return a iterator which yields object data.

        Args:
            object_name (str): Object name
        
        Returns:
            Iterator[bytes]: The object, as a byte stream.
        """
        pass
