# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""OCI-Compatible object store."""

from __future__ import annotations

import concurrent.futures
import os
import pathlib
import uuid
from tempfile import TemporaryDirectory
from typing import Callable, Optional, Union

from composer.utils.import_helpers import MissingConditionalImportError
from composer.utils.object_store.object_store import ObjectStore

__all__ = ['OCIObjectStore']


def _reraise_oci_errors(uri: str, e: Exception):
    try:
        import oci
    except ImportError as e:
        raise MissingConditionalImportError(
            conda_package='oci',
            extra_deps_group='oci',
            conda_channel='conda-forge',
        ) from e

    # If it's an oci service error with code: ObjectNotFound or status 404
    if isinstance(e, oci.exceptions.ServiceError):
        if e.status == 404:  # type: ignore
            if e.code == 'ObjectNotFound':  # type: ignore
                raise FileNotFoundError(f'Object {uri} not found. {e.message}') from e  # type: ignore
            if e.code == 'BucketNotFound':  # type: ignore
                raise ValueError(f'Bucket specified in {uri} not found. {e.message}') from e  # type: ignore
            raise FileNotFoundError(f'Object {uri} not found with no error code. {e.message}') from e  # type: ignore

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
            raise MissingConditionalImportError(
                conda_package='oci',
                extra_deps_group='oci',
                conda_channel='conda-forge',
            ) from e

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

            self.client = oci.object_storage.ObjectStorageClient(
                config=config,
                retry_strategy=oci.retry.DEFAULT_RETRY_STRATEGY,
            )
        except Exception as e:
            _reraise_oci_errors(self.get_uri(object_name=''), e)

        self.namespace = self.client.get_namespace().data  # pyright: ignore[reportOptionalMemberAccess]
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

        if response.status == 200:  # pyright: ignore[reportUnboundVariable, reportOptionalMemberAccess]
            data = response.data  # pyright: ignore[reportUnboundVariable, reportOptionalMemberAccess]
            return int(data.headers['Content-Length'])
        else:
            status = response.status  # pyright: ignore[reportUnboundVariable, reportOptionalMemberAccess]
            raise ValueError(f'OCI get_object was not successful with a {status} status code.')

    def upload_object(
        self,
        object_name: str,
        filename: Union[str, pathlib.Path],
        callback: Optional[Callable[[int, int], None]] = None,
    ):
        del callback
        try:
            self.upload_manager.upload_file(
                namespace_name=self.namespace,
                bucket_name=self.bucket,
                object_name=object_name,
                file_path=filename,
            )

        except Exception as e:
            _reraise_oci_errors(self.get_uri(object_name), e)

    def _download_part(self, object_name, filename, start_byte, end_byte, part_number):
        range_header = f'bytes={start_byte}-{end_byte}'
        tmp_part_path = os.path.join(filename, f'part-{part_number}-{uuid.uuid4()}.tmp')
        response = self.client.get_object(
            namespace_name=self.namespace,
            bucket_name=self.bucket,
            object_name=object_name,
            range=range_header,
        )
        with open(tmp_part_path, 'wb') as f:
            f.write(response.data.content)  # pyright: ignore[reportOptionalMemberAccess]
        return part_number, tmp_part_path

    def download_object(
        self,
        object_name: str,
        filename: Union[str, pathlib.Path],
        overwrite: bool = False,
        callback: Optional[Callable[[int, int], None]] = None,
        min_part_size: int = 128000000,
        num_parts: int = 10,
    ):
        del callback
        if os.path.exists(filename) and not overwrite:
            raise FileExistsError(f'The file at {filename} already exists and overwrite is set to False')

        dirname = os.path.dirname(filename)
        if dirname:
            os.makedirs(dirname, exist_ok=True)

        # Get the size of the object
        object_size = 0
        try:
            head_object_response = self.client.head_object(self.namespace, self.bucket, object_name)
            object_size = int(
                head_object_response.headers['content-length'],  # pyright: ignore[reportOptionalMemberAccess]
            )
        except Exception as e:
            _reraise_oci_errors(self.get_uri(object_name), e)

        # Calculate the part sizes
        num_parts_from_size = max(object_size // min_part_size, 1)
        num_parts = min(num_parts, num_parts_from_size)
        base_part_size, remainder = divmod(object_size, num_parts)
        part_sizes = [base_part_size] * num_parts
        for i in range(remainder):
            part_sizes[i] += 1
        part_sizes = [part_size for part_size in part_sizes if part_size > 0]

        with TemporaryDirectory(dir=dirname, prefix=f'{str(filename)}') as temp_dir:
            parts = []
            try:
                # Download parts in parallel
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    futures = []
                    start_byte = 0
                    for i, part_size in enumerate(part_sizes):
                        end_byte = start_byte + part_size - 1
                        futures.append(
                            executor.submit(self._download_part, object_name, temp_dir, start_byte, end_byte, i),
                        )
                        start_byte = end_byte + 1

                    for future in concurrent.futures.as_completed(futures):
                        parts.append(future.result())
                    parts = sorted(parts, key=lambda x: x[0])
            except Exception as e:
                _reraise_oci_errors(self.get_uri(object_name), e)

            # Combine parts
            tmp_path = os.path.join(temp_dir, f'{str(filename)}-{uuid.uuid4()}.tmp')
            with open(tmp_path, 'wb') as outfile:
                for i, part_file_name in parts:
                    with open(part_file_name, 'rb') as infile:
                        outfile.write(infile.read())

            if overwrite:
                os.replace(tmp_path, filename)
            else:
                os.rename(tmp_path, filename)

    def list_objects(self, prefix: Optional[str] = None) -> list[str]:
        if prefix is None:
            prefix = ''

        if self.prefix:
            prefix = f'{self.prefix}{prefix}'

        object_names = []
        next_start_with = None
        response_complete = False
        try:
            while not response_complete:
                response = self.client.list_objects(
                    namespace_name=self.namespace,
                    bucket_name=self.bucket,
                    prefix=prefix,
                    start=next_start_with,
                ).data  # pyright: ignore[reportOptionalMemberAccess]
                object_names.extend([obj.name for obj in response.objects])
                next_start_with = response.next_start_with
                if not next_start_with:
                    response_complete = True
        except Exception as e:
            _reraise_oci_errors(self.get_uri(prefix), e)

        return object_names
