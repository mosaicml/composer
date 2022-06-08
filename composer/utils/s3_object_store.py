# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Utility for uploading to and downloading from cloud object stores."""
import io
import os
import sys
import uuid
from typing import Any, Dict, Iterator, Optional, Union

import boto3
from botocore.config import Config

from composer.utils.iter_helpers import IteratorFileStream
from composer.utils.object_store import ObjectStore

__all__ = ["S3ObjectStore"]


class S3ObjectStore(ObjectStore):
    """Utility for uploading to and downloading from Amazon S3."""

    def __init__(self, 
            bucket: Optional[str] = None,
            s3_client_config: Optional[Dict[Any, Any]] = None,
            s3_transfer_config: Optional[Dict[Any, Any]] = None) -> None:
        self.bucket = bucket
        config = Config(**s3_client_config)
        self.client = boto3.client('s3', config=config)
        self.s3_transfer_config = s3_transfer_config

    def upload_object(self,
                      file_path: str,
                      object_name: str,
                      bucket: Optional[str] = None,
                      extra: Optional[Dict] = None):
        """Upload an object currently located on a disk.

        Args:
            file_path (str): Path to the object on disk.
            object_name (str): Object name (i.e. where the object will be stored in the container.)
            extra (Optional[Dict], optional): Extra attributes to pass to the underlying provider driver.
                (default: ``None``, which is equivalent to an empty dictionary)
        """
        try:
            self.client.upload_file(Filename=file_path, Bucket=bucket, Key=object_name, ExtraArgs=extra)
        except:
            raise Exception("Custom exception implemented later")

    def upload_object_via_stream(self,
                                 obj: Union[bytes, Iterator[bytes]],
                                 object_name: str,
                                 bucket: Optional[str] = None,
                                 extra: Optional[Dict] = None):
        """Upload an object.

        Args:
            obj (bytes | Iterator[bytes]): The object.
            object_name (str): Object name (i.e. where the object will be stored in the container.)
            extra (Optional[Dict], optional): Extra attributes to pass to the underlying provider driver.
                (default: ``None``)
        """
        if isinstance(obj, bytes):
            obj = iter(i.to_bytes(1, sys.byteorder) for i in obj)
        file_obj = io.BufferedReader(IteratorFileStream(obj))
        
        self.client.upload_fileobj(file_obj, Bucket=bucket, Key=object_name, ExtraArgs=extra)

    def download_object(self,
                        object_name: str,
                        destination_path: str,
                        bucket: Optional[str] = None):
        """Download an object to the specified destination path."""

        tmp_path = object_name + f".{uuid.uuid4()}.tmp"
        try:
            self.client.download_file(Bucket=bucket, Key=tmp_path, Filename=destination_path)
        except:
            raise Exception # custom exception to be be implemented later
        else:
            os.replace(tmp_path, destination_path)


    def download_object_as_stream(self, object_name: str, bucket: Optional[str] = None, chunk_size: Optional[int] = None):
        """Return a iterator which yields object data.

        Args:
            object_name (str): Object name.
            chunk_size (Optional[int], optional): Optional chunk size (in bytes).

        Returns:
            Iterator[bytes]: The object, as a byte stream.
        """
        
        obj = io.BytesIO()
        self.client.download_fileobj(Bucket=bucket, Key=object_name, Fileobj=obj)
        obj.seek(0)
        return obj
