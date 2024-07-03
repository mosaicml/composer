# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Google Cloud SDK - Compatible object store."""

from __future__ import annotations

import os
import pathlib
import uuid
from typing import Callable, Optional, Union

from composer.utils.import_helpers import MissingConditionalImportError
from composer.utils.object_store.object_store import ObjectStore

__all__ = ['GCSObjectStore']

BOTOCORE_CLIENT_ERROR_CODES = ('403', '404', 'NoSuchKey')


def _reraise_gcs_errors(uri: str, e: Exception):
    try:
        from google.api_core.exceptions import GatewayTimeout, NotFound

    except ImportError as import_exception:
        raise MissingConditionalImportError(
            conda_package='google-cloud-storage',
            extra_deps_group='google-cloud-storage',
            conda_channel='conda-forge',
        ) from import_exception

    # If it's a google service NotFound error
    if isinstance(e, NotFound):
        raise FileNotFoundError(f'Object {uri} not found.') from e

    # All clienterror (HTTP 4xx) responses
    elif isinstance(e, GatewayTimeout):
        raise ValueError(f'Time out when uploading/downloading {uri} using google cloud storage') from e

    # Otherwise just raise the original error.
    raise e


class GCSObjectStore(ObjectStore):
    """Utility for uploading to and downloading from a Google Cloud bucket using :mod:`google cloud storage sdk` with either HMAC or service account authentications. If both authentiations are available, this class will use service account authentication.

    .. warning::

        For best security practices, it is recommended to set credentials via environment variables or config files.

        See :ref:`guide to credentials <boto3:guide_credentials>` for more information.

    Args:
        bucket (str): The name of the Google Cloud bucket to upload to or download from.
        prefix (str, optional): The prefix to use when uploading to or downloading from the bucket. Default is an empty string.
    """

    def __init__(
        self,
        bucket: str,
        prefix: str = '',
    ) -> None:
        try:
            from google.auth import default as default_auth
            from google.auth.exceptions import DefaultCredentialsError
            from google.cloud.storage import Client
        except ImportError as e:
            raise MissingConditionalImportError('gcs', 'google.cloud.storage') from e

        # Format paths
        self.bucket_name = bucket.strip('/')
        self.prefix = prefix.strip('/')
        if self.prefix != '':
            self.prefix += '/'

        self.s3_object_store = None

        if 'GCS_KEY' in os.environ and 'GCS_SECRET' in os.environ:
            # Create a session and use it to make our client. Unlike Resources and Sessions,
            # clients are generally thread-safe.

            from composer.utils.object_store.s3_object_store import S3ObjectStore

            self.s3_object_store = S3ObjectStore(
                bucket=self.bucket_name,
                prefix=self.prefix,
                region_name='auto',
                endpoint_url='https://storage.googleapis.com',
                aws_access_key_id=os.environ['GCS_KEY'],
                aws_secret_access_key=os.environ['GCS_SECRET'],
            )
            self.client = None
            self.use_gcs_sdk = False
        else:
            try:
                credentials, _ = default_auth()
                self.client = Client(credentials=credentials)
                self.use_gcs_sdk = True
                try:
                    self.bucket = self.client.get_bucket(self.bucket_name, timeout=60)
                except Exception as e:
                    _reraise_gcs_errors(self.get_uri(object_name=''), e)
            except (DefaultCredentialsError, EnvironmentError) as e:
                raise ValueError(
                    f'No GCS_KEY/GCS_SECRET found and client construction failed with `{e}`. '
                    'Either set the environment variables `GCS_KEY` and `GCS_SECRET` or use any of '
                    'the methods in https://cloud.google.com/docs/authentication/external/set-up-adc '
                    'to set up Application Default Credentials. '
                    'See also https://docs.mosaicml.com/projects/mcli/en/latest/resources/secrets/gcp.html.',
                )

    def get_key(self, object_name: str) -> str:
        return f'{self.prefix}{object_name}'

    def get_uri(self, object_name: str) -> str:
        return f'gs://{self.bucket_name}/{self.get_key(object_name)}'

    def get_object_size(self, object_name: str) -> int:
        """Retrieves the size of an object stored in the cloud storage bucket.

        Args:
            object_name (str): The name of the object in the cloud storage bucket whose size is to be retrieved.

        Returns:
            int: The size of the object in bytes.

        Raises:
            FileNotFoundError: If the specified object does not exist in the cloud storage bucket.
            Exception: If an error occurs while trying to retrieve the object's size.
        """
        if not self.use_gcs_sdk:
            assert self.s3_object_store is not None
            return self.s3_object_store.get_object_size(object_name)

        from google.cloud.storage import Blob

        key = self.get_key(object_name)
        blob_exists = Blob(bucket=self.bucket, name=key).exists(self.client)
        if not blob_exists:
            raise FileNotFoundError(f'{object_name} not found in {self.bucket_name}')
        blob = None
        try:
            key = self.get_key(object_name)
            blob = self.bucket.get_blob(key)
        except Exception as e:
            _reraise_gcs_errors(self.get_uri(object_name), e)

        if blob is None or blob.size is None:
            return -1
        return blob.size  # size in bytes

    def upload_object(
        self,
        object_name: str,
        filename: Union[str, pathlib.Path],
        callback: Optional[Callable[[int, int], None]] = None,
    ):
        """Uploads a file to the cloud storage bucket.

        Args:
            object_name (str, optional): The destination path in the cloud storage bucket where the file will be saved.
                If not provided or an empty string is given, the file will be uploaded to the root of the bucket with the same
                name as the source file. Default is an empty string.
            filename (Union[str, pathlib.Path]): The path to the local file
            callback: optional
        """
        if not self.use_gcs_sdk:
            assert self.s3_object_store is not None
            return self.s3_object_store.upload_object(object_name, filename, callback)

        if callback is not None:
            raise ValueError('callback is not supported in gcs upload_object()')
        from google.cloud.storage.retry import DEFAULT_RETRY
        src = filename
        dest = object_name
        dest = str(src) if dest == '' else dest
        blob = self.bucket.blob(self.get_key(dest))
        blob.upload_from_filename(src, retry=DEFAULT_RETRY)  # pyright: ignore[reportGeneralTypeIssues]

    def download_object(
        self,
        object_name: str,
        filename: Union[str, pathlib.Path],
        overwrite: bool = False,
        callback: Optional[Callable[[int, int], None]] = None,
    ):
        """Downloads an object from the specified source in the cloud storage bucket and saves it to the given destination.

        Args:
            object_name (str): The path to the object in the cloud storage bucket that needs to be downloaded.
            filename (Union[str, pathlib.Path]): The destination path where the object will be saved locally. It can be a
                string representing the file path or a pathlib.Path object.
            overwrite (bool, optional): If set to True, the function will overwrite the destination file if it already
                exists. If set to False, and the destination file exists, a FileExistsError will be raised. Default is False.
            callback (Callable[[int, int], None], optional): A callback function that can be used to track the progress of
                the download. It takes two integer arguments - the number of bytes downloaded and the total size of the
                object. Default is None. Unused for GCSObjectStore.

        Raises:
            FileExistsError: If the destination file already exists and the `overwrite` parameter is set to False.
        """
        if not self.use_gcs_sdk:
            assert self.s3_object_store is not None
            return self.s3_object_store.download_object(object_name, filename, overwrite, callback)
        dest = filename
        src = object_name

        if os.path.exists(dest) and not overwrite:
            raise FileExistsError(f'The file at {dest} already exists and overwrite is set to False.')

        dirname = os.path.dirname(dest)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        tmp_path = str(dest) + f'.{uuid.uuid4()}.tmp'

        try:
            try:
                blob = self.bucket.blob(self.get_key(src))
                blob.download_to_filename(tmp_path)
            except Exception as e:
                _reraise_gcs_errors(self.get_uri(src), e)
        except:
            # Make a best effort attempt to clean up the temporary file
            try:
                os.remove(tmp_path)
            except OSError:
                pass
            raise
        else:
            if overwrite:
                os.replace(tmp_path, dest)
            else:
                os.rename(tmp_path, dest)

    def list_objects(self, prefix: Optional[str] = None) -> list[str]:
        if not self.use_gcs_sdk:
            assert self.s3_object_store is not None
            return self.s3_object_store.list_objects(prefix)
        if prefix is None:
            prefix = ''
        prefix = self.get_key(prefix)

        objects = []
        try:
            objects = self.bucket.list_blobs(prefix=prefix)
        except Exception as e:
            _reraise_gcs_errors(prefix, e)

        return [ob.name for ob in objects]
