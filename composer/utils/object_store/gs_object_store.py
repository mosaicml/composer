# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Google Cloud SDK - Compatible object store."""

from __future__ import annotations

import os
import pathlib
import urllib
import uuid
from google.cloud.storage import Blob, Bucket, Client
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

    print("Reraising exception: {e.message}")

    # If it's a google service NotFound error
    if isinstance(e, google.api_core.exceptions.NotFound):
        raise FileNotFoundError(f'Object {uri} not found. {e.message}') from e

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
        remote_dir (str, optional): Required. The URL to an Gooogle Cloud Storage object, formatted as scheme://bucket/path
    """

    def __init__(
        self,
        gs_root_dir: str #  = 'gs://mosaicml-composer-tests/streaming/',
    ) -> None:
        try:
            import boto3
        except ImportError as e:
            raise MissingConditionalImportError('streaming', 'boto3') from e

        if 'GOOGLE_APPLICATION_CREDENTIALS' in os.environ:
            service_account_path = os.environ['GOOGLE_APPLICATION_CREDENTIALS']
            self.client = storage.Client.from_service_account_json(service_account_path)
        else:
            raise ValueError(f'GOOGLE_APPLICATION_CREDENTIALS needs to be set for ' +
                             f'service level accounts ')

        obj = urllib.parse.urlparse(gs_root_dir)
        if obj.netloc == '':
            raise ValueError("remote_dir doesn't have a valid format")

        self.bucket_name = obj.netloc
        self.prefix = obj.path.lstrip('/')

        try:
            self.bucket = self.client.get_bucket(self.bucket_name, timeout=10.0) # Bucket(self.client, obj.netloc)
        except Exception as e:
            _reraise_gs_errors(obj.netloc, e)

    def get_key(self, object_name: str) -> str:
        return f'{self.prefix}{object_name}'

    def get_uri(self, object_name: str) -> str:
        return f'gs://{self.bucket_name}/{self.get_key(object_name)}'

    def get_object_size(self, object_name: str) -> int:
        key = self.get_key(object_name)
        stats = storage.Blob(bucket=self.bucket, name=key).exists(self.client)
        if not stats:
            raise FileNotFoundError(f'{object_name} not found in {self.bucket_name}')
        try:
            key = self.get_key(object_name)
            blob = self.bucket.get_blob(key)
        except Exception as e:
            _reraise_gs_errors(self.get_uri(object_name), e)

        return blob.size  # size in byte

    def upload_blob(self,
                    src: Union[str, pathlib.Path],
                    dest: Union[str, pathlib.Path] = None,
                    callback: Optional[Callable[[int, int], None]] = None,
                    generation_match_precondition = 0):

        del callback
        """Uploads a file to the bucket.
           By default, if_generation_not_match = 0 makes the operation succeed only if there is a live version of the blob.
        """

        dest = str(src) if dest is None  else dest
        blob = self.bucket.blob(self.get_key(dest))
        blob.upload_from_filename(src) # , if_generation_match=generation_match_precondition)

        print(f'File {src} uploaded to {dest}.')

    def download_blob(
        self,
        src: str,
        dest: Union[str, pathlib.Path],
        overwrite: bool = False,
        callback: Optional[Callable[[int, int], None]] = None,
    ):

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
                _reraise_gs_errors(self.get_uri(src), e)
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
