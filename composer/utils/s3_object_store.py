# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Utility for uploading to and downloading from cloud object stores."""
import io
from typing import Any, Dict, Iterator, Optional, Union
import boto3
from botocore.config import Config
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

        if bucket is None:
            if self.bucket is None:
                raise ValueError("No S3 bucket specified")
            bucket = self.bucket

        self.client.upload_file(Filename=file_path, Bucket=bucket, Key=object_name, ExtraArgs=extra)

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

        if bucket is None:
            if self.bucket is None:
                raise ValueError("No S3 bucket specified")
            bucket = self.bucket
        
        # add handling for other bytes-like types
        if not isinstance(obj, bytes):
            raise NotImplementedError("S3ObjectStore.upload_object_via_stream only takes a bytes object.")
        
        obj = io.BytesIO(obj)

        # if isinstance(obj, bytes):
        #     obj = iter(i.to_bytes(1, sys.byteorder) for i in obj)
        
        self.client.upload_fileobj(obj, Bucket=bucket, Key=object_name, ExtraArgs=extra)

    def download_object(self,
                        object_name: str,
                        destination_path: str,
                        bucket: Optional[str] = None):
        """Download an object to the specified destination path.
        """

        if bucket is None:
            if self.bucket is None:
                raise ValueError("No S3 bucket specified")
            bucket = self.bucket

        self.client.download_file(Bucket=bucket, Key=object_name, Filename=destination_path)

    def download_object_as_stream(self, object_name: str, bucket: Optional[str] = None, chunk_size: Optional[int] = None):
        """Return a iterator which yields object data.

        Args:
            object_name (str): Object name.
            chunk_size (Optional[int], optional): Optional chunk size (in bytes).

        Returns:
            Iterator[bytes]: The object, as a byte stream.
        """

        if bucket is None:
            if self.bucket is None:
                raise ValueError("No S3 bucket specified")
            bucket = self.bucket
        
        obj = io.BytesIO()
        self.client.download_fileobj(Bucket=bucket, Key=object_name, Fileobj=obj)
        obj.seek(0)
        return obj
