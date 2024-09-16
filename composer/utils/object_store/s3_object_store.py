# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""S3-Compatible object store."""

from __future__ import annotations

import os
import pathlib
import uuid
from typing import Any, Callable, Optional, Union

from composer.utils.import_helpers import MissingConditionalImportError
from composer.utils.object_store.object_store import ObjectStore

__all__ = ['S3ObjectStore']

_NOT_FOUND_CODES = ('403', '404', 'NoSuchKey')


def _ensure_not_found_errors_are_wrapped(uri: str, e: Exception):
    import botocore.exceptions
    if isinstance(e, botocore.exceptions.ClientError):
        # error: Member "response" is unknown (reportGeneralTypeIssues)
        if e.response['Error']['Code'] in _NOT_FOUND_CODES:  # type: ignore
            raise FileNotFoundError(f'Object {uri} not found') from e
    raise e


class S3ObjectStore(ObjectStore):
    """Utility for uploading to and downloading from an S3-compatible bucket using :mod:`boto3`.

    .. note::

        Since :mod:`boto3` has automatic retry handling, transient errors are not
        re-raised with :exc:`.ObjectStoreTransientError`. By default, the "Legacy retry mode" makes at most five
        attempts for operations that failed due to transient errors. This setting can be customized via the
        ``retries`` key in ``client_config``.

        See the :ref:`guide to retries <boto3:guide_retries>` for additional information.


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
        region_name: Optional[str] = None,
        endpoint_url: Optional[str] = None,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        aws_session_token: Optional[str] = None,
        client_config: Optional[dict[Any, Any]] = None,
        transfer_config: Optional[dict[Any, Any]] = None,
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
        if 'retries' not in client_config:
            client_config['retries'] = {'mode': 'adaptive'}
        config = Config(**client_config)
        if 'S3_ENDPOINT_URL' in os.environ and endpoint_url is None:
            endpoint_url = os.environ['S3_ENDPOINT_URL']

        self.client = boto3.Session().client(
            's3',
            config=config,
            region_name=region_name,
            endpoint_url=endpoint_url,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            aws_session_token=aws_session_token,
        )
        if transfer_config is None:
            transfer_config = {}
        self.transfer_config = TransferConfig(**transfer_config)

    def get_uri(self, object_name: str) -> str:
        return f's3://{self.bucket}/{self.get_key(object_name)}'

    def get_key(self, object_name: str) -> str:
        return f'{self.prefix}{object_name}'

    def get_object_size(self, object_name: str) -> int:
        obj = {'ContentLength': -1}
        try:
            obj = self.client.get_object(Bucket=self.bucket, Key=self.get_key(object_name))
        except Exception as e:
            _ensure_not_found_errors_are_wrapped(self.get_uri(object_name), e)
        return obj['ContentLength']

    def upload_object(
        self,
        object_name: str,
        filename: Union[str, pathlib.Path],
        callback: Optional[Callable[[int, int], None]] = None,
        **kwargs,
    ):
        try:
            from boto3.s3.transfer import S3Transfer
        except ImportError as e:
            raise MissingConditionalImportError('streaming', 'boto3') from e

        file_size = os.path.getsize(filename)
        cb_wrapper = None if callback is None else lambda bytes_transferred: callback(bytes_transferred, file_size)

        # Validate kwargs. Use env var for Canned ACL if present and one has not been passed in.
        if len(kwargs) == 0:
            if 'S3_CANNED_ACL' in os.environ:
                kwargs['ExtraArgs'] = {'ACL': os.environ['S3_CANNED_ACL']}
        else:
            if len(kwargs) > 1 or 'ExtraArgs' not in kwargs or not isinstance(kwargs['ExtraArgs'], dict):
                raise ValueError('S3ObjectStore.upload_object only supports an additional ExtraArgs dictionary.')
            for key in kwargs['ExtraArgs']:
                if key not in S3Transfer.ALLOWED_UPLOAD_ARGS:
                    raise ValueError(f'{key} is not an allowed upload argument.')
            if 'S3_CANNED_ACL' in os.environ and 'ACL' not in kwargs['ExtraArgs']:
                kwargs['ExtraArgs']['ACL'] = os.environ['S3_CANNED_ACL']

        self.client.upload_file(
            Bucket=self.bucket,
            Key=self.get_key(object_name),
            Filename=filename,
            Callback=cb_wrapper,
            Config=self.transfer_config,
            **kwargs,
        )

    def download_object(
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

        if callback is None:
            cb_wrapper = None
        else:
            file_size = self.get_object_size(object_name)
            cb_wrapper = lambda bytes_transferred: callback(bytes_transferred, file_size)

        try:
            try:
                self.client.download_file(
                    Bucket=self.bucket,
                    Key=self.get_key(object_name),
                    Filename=tmp_path,
                    Callback=cb_wrapper,
                    Config=self.transfer_config,
                )
            except Exception as e:
                _ensure_not_found_errors_are_wrapped(self.get_uri(object_name), e)
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

    def list_objects(self, prefix: Optional[str] = None) -> list[str]:
        if prefix is None:
            prefix = ''

        if self.prefix:
            prefix = f'{self.prefix}{prefix}'

        paginator = self.client.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=self.bucket, Prefix=prefix)
        try:
            return [obj['Key'] for page in pages for obj in page['Contents']]
        except KeyError:
            return []
