# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""OCI-Compatible object store."""

from __future__ import annotations

import os
import pathlib
import uuid
from typing import Callable, Optional, Union

from composer.utils.import_helpers import MissingConditionalImportError
from composer.utils.object_store.object_store import ObjectStore

__all__ = ['OCIObjectStore']


def _reraise_oci_errors(uri: str, e: Exception):
    try:
        import oci
    except ImportError as e:
        raise MissingConditionalImportError(conda_package='oci', extra_deps_group='oci',
                                            conda_channel='conda-forge') from e

    # If it's an oci service error with code: ObjectNotFound or status 404
    if isinstance(e, oci.exceptions.ServiceError):
        if e.status == 404:  # type: ignore
            if e.code == 'ObjectNotFound':  # type: ignore
                raise FileNotFoundError(f'Object {uri} not found. {e.message}') from e  # type: ignore
            if e.code == 'BucketNotFound':  # type: ignore
                raise ValueError(f'Bucket specified in {uri} not found. {e.message}') from e  # type: ignore
            raise e

    # Client errors
    if isinstance(e, oci.exceptions.ClientError):
        raise ValueError(f'Error with using your OCI config file for uri {uri}') from e
    if isinstance(e, oci.exceptions.MultipartUploadError):
        raise ValueError(f'Error when uploading {uri} using OCI parallelized uploading') from e

    # Otherwise just raise the original error.
    raise e


class OCIObjectStore(ObjectStore):
    """Utility for uploading to and downloading from an OCI bucket.

    Args:
        bucket (str): The bucket name.
        prefix (str): A path prefix such as `folder/subfolder/` to prepend to object names. Defaults to ''.
    """

    def __init__(
        self,
        bucket: str,
        prefix: str = '',
    ) -> None:
        try:
            import oci
        except ImportError as e:
            raise MissingConditionalImportError(conda_package='oci',
                                                extra_deps_group='oci',
                                                conda_channel='conda-forge') from e

        # Format paths
        self.bucket = bucket.strip('/')
        self.prefix = prefix.strip('/')
        if self.prefix != '':
            self.prefix += '/'

        try:
            if 'OCI_CONFIG_FILE' in os.environ:
                config = oci.config.from_file(os.environ['OCI_CONFIG_FILE'])
            else:
                config = oci.config.from_file()

            self.client = oci.object_storage.ObjectStorageClient(config=config,
                                                                 retry_strategy=oci.retry.DEFAULT_RETRY_STRATEGY)
        except Exception as e:
            _reraise_oci_errors(self.get_uri(object_name=''), e)

        self.namespace = self.client.get_namespace().data
        self.upload_manager = oci.object_storage.UploadManager(self.client)

    def get_uri(self, object_name: str) -> str:
        return f'oci://{self.bucket}/{object_name}'

    def get_object_size(self, object_name: str) -> int:
        try:
            response = self.client.get_object(
                namespace_name=self.namespace,
                bucket_name=self.bucket,
                object_name=object_name,
            )
        except Exception as e:
            _reraise_oci_errors(self.get_uri(object_name), e)

        if response.status == 200:
            return int(response.data.headers['Content-Length'])
        else:
            raise ValueError(f'OCI get_object was not successful with a {response.status} status code.')

    def upload_object(
        self,
        object_name: str,
        filename: Union[str, pathlib.Path],
        callback: Optional[Callable[[int, int], None]] = None,
    ):
        del callback
        try:
            self.upload_manager.upload_file(namespace_name=self.namespace,
                                            bucket_name=self.bucket,
                                            object_name=object_name,
                                            file_path=filename)

        except Exception as e:
            _reraise_oci_errors(self.get_uri(object_name), e)

    def download_object(
        self,
        object_name: str,
        filename: Union[str, pathlib.Path],
        overwrite: bool = False,
        callback: Optional[Callable[[int, int], None]] = None,
    ):
        del callback
        if os.path.exists(filename) and not overwrite:
            raise FileExistsError(f'The file at {filename} already exists and overwrite is set to False')
        tmp_path = str(filename) + f'.{uuid.uuid4()}.tmp'

        try:
            response = self.client.get_object(
                namespace_name=self.namespace,
                bucket_name=self.bucket,
                object_name=object_name,
            )
        except Exception as e:
            _reraise_oci_errors(self.get_uri(object_name), e)

        with open(tmp_path, 'wb') as f:
            f.write(response.data.content)

        if overwrite:
            os.replace(tmp_path, filename)
        else:
            os.rename(tmp_path, filename)
