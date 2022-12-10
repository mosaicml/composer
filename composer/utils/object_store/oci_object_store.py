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

        if 'OCI_CONFIG_FILE' in os.environ:
            config = oci.config.from_file(os.environ['OCI_CONFIG_FILE'])
        else:
            config = oci.config.from_file()

        self.client = oci.object_storage.ObjectStorageClient(config=config,
                                                             retry_strategy=oci.retry.DEFAULT_RETRY_STRATEGY)
        self.namespace = self.client.get_namespace().data
        self.upload_manager = oci.object_storage.UploadManager(self.client)

    def get_uri(self, object_name: str) -> str:
        return f'oci://{self.bucket}/{object_name}'

    def get_object_size(self, object_name: str) -> int:
        response = self.client.get_object(
            namespace_name=self.namespace,
            bucket_name=self.bucket,
            object_name=object_name,
        )
        if response.status == 200:
            return int(response.data.headers['Content-Length'])
        else:
            raise FileNotFoundError(f'Unable to locate oci://{self.bucket}@{self.namespace}/{object_name}')

    def upload_object(
        self,
        object_name: str,
        filename: Union[str, pathlib.Path],
        callback: Optional[Callable[[int, int], None]] = None,
    ):
        del callback

        self.upload_manager.upload_file(namespace_name=self.namespace,
                                        bucket_name=self.bucket,
                                        object_name=object_name,
                                        file_path=filename)

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

        response = self.client.get_object(
            namespace_name=self.namespace,
            bucket_name=self.bucket,
            object_name=object_name,
        )

        with open(tmp_path, 'wb') as f:
            f.write(response.data.content)

        if overwrite:
            os.replace(tmp_path, filename)
        else:
            os.rename(tmp_path, filename)
