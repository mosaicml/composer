# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Utility for uploading to and downloading from cloud object stores."""
import os
import sys
import tempfile
import uuid
import io
from typing import Any, Dict, Iterator, Optional, Union
import boto3
from botocore.config import Config
from composer.utils.object_store import ObjectStore

__all__ = ["S3ObjectStore"]


class S3ObjectStore(ObjectStore):
    """Utility for uploading to and downloading from Amazon S3.
    """

    def __init__(self, 
            container: str,
            s3_client_config = Dict[Any, Any],
            s3_transfer_config = Dict[Any, Any]) -> None:
        self._container = container
        config = Config(**s3_client_config)
        self.client = boto3.client('s3', config=config)

        self.s3_transfer_config = s3_transfer_config

    @property
    def container_name(self):
        """The name of the object storage container."""
        return self._container.name

    def upload_object(self,
                      file_path: str,
                      object_name: str,
                      verify_hash: bool = True,
                      extra: Optional[Dict] = None,
                      headers: Optional[Dict[str, str]] = None):
        """Upload an object currently located on a disk.

        .. seealso:: :meth:`libcloud.storage.base.StorageDriver.upload_object`.

        Args:
            file_path (str): Path to the object on disk.
            object_name (str): Object name (i.e. where the object will be stored in the container.)
            verify_hash (bool, optional): Whether to verify hashes (default: ``True``)
            extra (Optional[Dict], optional): Extra attributes to pass to the underlying provider driver.
                (default: ``None``, which is equivalent to an empty dictionary)
            headers (Optional[Dict[str, str]], optional): Additional request headers, such as CORS headers.
                (defaults: ``None``, which is equivalent to an empty dictionary)
        """
        del headers, verify_hash # used in the libcloud interface

        self.client.upload_file(filename=file_path, bucket=self.container_name, key=object_name, ExtraArgs=extra)

    def upload_object_via_stream(self,
                                 obj: Union[bytes, Iterator[bytes]],
                                 object_name: str,
                                 extra: Optional[Dict] = None,
                                 headers: Optional[Dict[str, str]] = None):
        """Upload an object.

        .. seealso:: :meth:`libcloud.storage.base.StorageDriver.upload_object_via_stream`.

        Args:
            obj (bytes | Iterator[bytes]): The object.
            object_name (str): Object name (i.e. where the object will be stored in the container.)
            verify_hash (bool, optional): Whether to verify hashes (default: ``True``)
            extra (Optional[Dict], optional): Extra attributes to pass to the underlying provider driver.
                (default: ``None``)
            headers (Optional[Dict[str, str]], optional): Additional request headers, such as CORS headers.
                (defaults: ``None``)
        """
        del headers
        
        # add handling for other bytes-like types
        if not isinstance(obj, bytes):
            raise NotImplementedError("S3ObjectStore.upload_object_via_stream only takes a bytes object.")
        
        obj = io.BytesIO(obj)

        # if isinstance(obj, bytes):
        #     obj = iter(i.to_bytes(1, sys.byteorder) for i in obj)
        
        self.client.upload_fileobj(obj, bucket=self.container_name, key=object_name, ExtraArgs=extra)

    def download_object(self,
                        object_name: str,
                        destination_path: str,
                        overwrite_existing: bool = False,
                        delete_on_failure: bool = True):
        """Download an object to the specified destination path.
        """
        self.client.download_file(bucket=self.container_name, key=object_name, filename=destination_path)

    def download_object_as_stream(self, object_name: str, chunk_size: Optional[int] = None):
        """Return a iterator which yields object data.

        Args:
            object_name (str): Object name.
            chunk_size (Optional[int], optional): Optional chunk size (in bytes).

        Returns:
            Iterator[bytes]: The object, as a byte stream.
        """
        obj = io.BytesIO()
        self.client.download_fileobj(Bucket=self.container_name, Key=object_name, Fileobj=obj)
        obj.seek(0)
        return obj
