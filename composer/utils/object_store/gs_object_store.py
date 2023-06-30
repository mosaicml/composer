# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Google Cloud SDK - Compatible object store."""

from __future__ import annotations

import os
import pathlib
import uuid
from typing import Any, Callable, Dict, Optional, Union

from composer.utils.import_helpers import MissingConditionalImportError
from composer.utils.object_store.object_store import ObjectStore

__all__ = ['GsObjectStore']

_NOT_FOUND_CODES = ('403', '404', 'NoSuchKey')

def _reraise_gs_errors(uri: str, e: Exception):
    try:
        import google.cloud
    except ImportError as e:
        raise MissingConditionalImportError(conda_package='google-cloud-storage', extra_deps_group='google-cloud-storage',
                                            conda_channel='conda-forge') from e

    # If it's an google service NotFound error
    if isinstance(e, google.api_core.exceptions.NotFound):
        raise ValueError(f'Bucket or object in {uri} not found. {e.message}') from e

    # All clienterror (HTTP 4xx) responses
    if isinstance(e,  google.api_core.exceptions.ClientError):
        raise ValueError(f'Error with using google cloud storage for uri {uri}') from e
    if isinstance(e, google.api_core.exceptions.GatewayTimeout):
        raise ValueError(f'Time out when uploading/downloading {uri} using google cloud storage') from e

    # Otherwise just raise the original error.
    raise e



class GsObjectStore(ObjectStore):
    """Utility for uploading to and downloading from an Google Cloud bucket using :mod:`google cloud sdk`.

    .. warning::

        For best security practices, it is recommended to set credentials via environment variables or config
        files. **Never hard-code credentials** via the ``aws_access_key_id``, ``aws_secret_access_key``, or
        ``aws_session_token`` arguments.

        See :ref:`guide to credentials <boto3:guide_credentials>` for more information.

    Args:
        bucket (str): The bucket name.
        prefix (str): A path prefix such as `folder/subfolder/` to prepend to object names. Defaults to ''.
        region_name (str, optional): The region name. Must be specified if not available in
            a config file or environment variables. Defaults to None.
        endpoint_url (str, optional): The URL to an S3-Compatible object store. Must be specified if using something
            other than Amazon S3, like Google Cloud Storage. Defaults to None.
        aws_access_key_id (str, optional): The access key id. Must be specified if not available in
            a config file or environment variables. Defaults to None.
        aws_secret_access_key (str, optional): The secret access key. Must be specified if not available in
            a config file or environment variables. Defaults to None.
        aws_session_token (str, optional): The session token. Must be specified if not available in
            a config file or environment variables. Defaults to None.
        client_config (dict, optional): Kwargs dictionary for :class:`botocore.config.Config`. Defaults to None.
        transfer_config (dict, optional): Kwargs dictionary for :class:`boto3.s3.transfer.TransferConfig`. Defaults to None.
    """

    def __init__(
        self,
        bucket: str,
        prefix: str = '',
        region_name: Optional[str] = 'auto',
        endpoint_url: Optional[str] = 'https://storage.googleapis.com',
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        aws_session_token: Optional[str] = None,
        client_config: Optional[Dict[Any, Any]] = None,
        transfer_config: Optional[Dict[Any, Any]] = None,
    ) -> None:
        try:
            import boto3
            from boto3.s3.transfer import TransferConfig
            from botocore.config import Config
        except ImportError as e:
            raise MissingConditionalImportError('streaming', 'boto3') from e

        # Format paths
        self.bucket = bucket.strip('/')
        self.prefix = prefix.strip('/')
        if self.prefix:
            self.prefix += '/'

        if client_config is None:
            client_config = {}
        config = Config(**client_config)

        if aws_secret_access_key is not None and aws_access_key_id is not None:
            # Create a session and use it to make our client. Unlike Resources and Sessions,
            # clients are generally thread-safe.

            import boto3
            session = boto3.session.Session()
            self.client = session.client('s3',
                                         region_name=region_name,
                                         endpoint_url=endpoint_url,
                                         aws_access_key_id=aws_access_key_id,
                                         aws_secret_access_key=aws_secret_access_key)

            return

        from google.cloud import storage
        self.client = storage.Client()


    def get_uri(self, object_name: str) -> str:
        return f's3://{self.bucket}/{self.get_key(object_name)}'

    def get_key(self, object_name: str) -> str:
        return f'{self.prefix}{object_name}'

    def get_object_size(self, object_name: str) -> int:
        try:
            bucket = self.client.get_bucket(self.bucket)
            blob = bucket.get_blob(object_name)
        except Exception as e:
            _reraise_gs_errors(self.get_uri(object_name), e)
        return blob.size # size in byte

    def upload_blob(
            self,
            bucket_name: str,
            source_file_name: Union[str, pathlib.Path],
            destination_blob_name: str = None,
            callback: Option[Callable[[int, int], None]] = None):

        del callback
        """Uploads a file to the bucket."""

        bucket = self.client.bucket(bucket_name)
        destination_blob_name = str(source_file_name) if destination_blob_name is None else destination_blob_name
        blob = bucket.blob(destination_blob_name)

        generation_match_precondition = 0

        blob.upload_from_filename(source_file_name, if_generation_match=generation_match_precondition)

        print(
            f"File {source_file_name} uploaded to {destination_blob_name}."
        )

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
                bucket = self.client.bucket(self.bucket_name)
                blob = bucket.blob(source_blob_name)
                blob.download_to_filename(tmp_path)
            except Exception as e:
                _reraise_gcs_errors(get_uri(object_name), e)
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


