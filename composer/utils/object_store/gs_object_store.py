# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Google Cloud SDK - Compatible object store."""

from __future__ import annotations

import os
import pathlib
import uuid
from typing import Callable, Optional, Union

from google.cloud import storage

from composer.utils.import_helpers import MissingConditionalImportError
from composer.utils.object_store.object_store import ObjectStore

__all__ = ['GsObjectStore']

_NOT_FOUND_CODES = ('403', '404', 'NoSuchKey')


def _reraise_gs_errors(uri: str, e: Exception):
    try:
        import google
    except ImportError as e:
        raise MissingConditionalImportError(conda_package='google-cloud-storage',
                                            extra_deps_group='google-cloud-storage',
                                            conda_channel='conda-forge') from e

    # If it's an google service NotFound error
    if isinstance(e, google.api_core.exceptions.NotFound):
        raise ValueError(f'Bucket or object in {uri} not found. {e.message}') from e

    # All clienterror (HTTP 4xx) responses
    if isinstance(e, google.api_core.exceptions.ClientError):
        raise ValueError(f'Error with using google cloud storage for uri {uri}') from e
    if isinstance(e, google.api_core.exceptions.GatewayTimeout):
        raise ValueError(f'Time out when uploading/downloading {uri} using google cloud storage') from e

    # Otherwise just raise the original error.
    raise e


class GsObjectStore(ObjectStore):
    """Utility for uploading to and downloading from an Google Cloud bucket using :mod:`google cloud storage sdk` with either HMAC and service account authentications. If both authentiations are available, use service account authentication.

    .. warning::

        For best security practices, it is recommended to set credentials via environment variables or config files.

        See :ref:`guide to credentials <boto3:guide_credentials>` for more information.

    Args:
        bucket (str): The bucket name.
        prefix (str): A path prefix such as `folder/subfolder/` to prepend to object names. Defaults to ''.
        region_name (str, optional): The region name. Must be specified if not available in
            a config file or environment variables. Defaults to None.
        endpoint_url (str, optional): The URL to an S3-Compatible object store. Must be specified if using something
            other than Amazon S3, like Google Cloud Storage. Defaults to None.
    """

    def __init__(
        self,
        bucket: str,
        prefix: str = '',
        region_name: Optional[str] = 'auto',
        endpoint_url: Optional[str] = 'https://storage.googleapis.com',
    ) -> None:
        try:
            import boto3
        except ImportError as e:
            raise MissingConditionalImportError('streaming', 'boto3') from e

        # Format paths
        self.bucket = bucket.strip('/')
        self.prefix = prefix.strip('/')
        if self.prefix:
            self.prefix += '/'

        if 'GOOGLE_APPLICATION_CREDENTIALS' in os.environ:
            service_account_path = os.environ['GOOGLE_APPLICATION_CREDENTIALS']
            self.client = storage.Client.from_service_account_json(service_account_path)

        elif 'GCS_KEY' in os.environ and 'GCS_SECRET' in os.environ:
            # Create a session and use it to make our client. Unlike Resources and Sessions,
            # clients are generally thread-safe.

            import boto3
            session = boto3.session.Session()
            self.client = session.client('s3',
                                         region_name=region_name,
                                         endpoint_url=endpoint_url,
                                         aws_access_key_id=os.environ['GCS_KEY'],
                                         aws_secret_access_key=os.environ['GCS_SECRET'])
        else:
            raise ValueError(f'Either GOOGLE_APPLICATION_CREDENTIALS needs to be set for ' +
                             f'service level accounts or GCS_KEY and GCS_SECRET needs to ' +
                             f'be set for HMAC authentication')

    def get_uri(self, object_name: str) -> str:
        return f'gs://{self.bucket}/{self.get_key(object_name)}'

    def get_key(self, object_name: str) -> str:
        return f'{self.prefix}{object_name}'

    def get_object_size(self, object_name: str) -> int:
        try:
            bucket = self.client.get_bucket(self.bucket)
            blob = bucket.get_blob(object_name)
        except Exception as e:
            _reraise_gs_errors(self.get_uri(object_name), e)
        return blob.size  # size in byte

    def upload_blob(self,
                    bucket_name: str,
                    source_file_name: Union[str, pathlib.Path],
                    destination_blob_name: str = '',
                    callback: Optional[Callable[[int, int], None]] = None):

        del callback
        """Uploads a file to the bucket."""

        bucket = self.client.bucket(bucket_name)
        destination_blob_name = str(source_file_name) if destination_blob_name is None else destination_blob_name
        blob = bucket.blob(destination_blob_name)

        generation_match_precondition = 0

        blob.upload_from_filename(source_file_name, if_generation_match=generation_match_precondition)

        print(f'File {source_file_name} uploaded to {destination_blob_name}.')

    def download_blob(
        self,
        object_name: str,
        filename: Union[str, pathlib.Path],
        overwrite: bool = False,
        callback: Optional[Callable[[int, int], None]] = None,
    ):

        if os.path.exists(filename) and not overwrite:
            raise FileExistsError(f'The file at {filename} already exists and overwrite is set to False.')

        dirname = os.path.dirname(filename)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        tmp_path = str(filename) + f'.{uuid.uuid4()}.tmp'

        try:
            try:
                bucket = self.client.bucket(self.bucket)
                blob = bucket.blob(object_name)
                blob.download_to_filename(tmp_path)
            except Exception as e:
                _reraise_gs_errors(self.get_uri(object_name), e)
        except:
            # Make a best effort attempt to clean up the temporary file
            try:
                os.remove(tmp_path)
            except OSError:
                pass
            raise
        else:
            if overwrite:
                os.replace(tmp_path, filename)
            else:
                os.rename(tmp_path, filename)
