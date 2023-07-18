from __future__ import annotations

import logging
import os
import pathlib
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional
from urllib.parse import urlparse

import requests

from composer.utils.import_helpers import MissingConditionalImportError
from composer.utils.object_store.object_store import ObjectStore
from composer.utils.object_store.s3_object_store import S3ObjectStore

log = logging.getLogger(__name__)


class Operation(Enum):
    READ_VOLUME = 1
    WRITE_VOLUME = 2


@dataclass
class S3Credentials:
    bucket: str
    prefix: str
    aws_access_key_id: str
    aws_session_token: str
    aws_secret_access_key: str


class UCVolumeObjectStore(ObjectStore):
    """
    Utility for uploading and downloading models from Unity Catalog Volumes
    """

    def __init__(
        self,
        full_name: str,
        databricks_host_name: Optional[str] = None,
        databricks_token: Optional[str] = None,
    ) -> None:
        try:
            import databricks
        except ImportError as e:
            raise MissingConditionalImportError('databricks') from e

        if not databricks_host_name:
            databricks_host_name = os.environ['DATABRICKS_HOST']

        # sanitize the host name to remove the workspace id
        parse_result = urlparse(databricks_host_name)
        self.databricks_host_name = f'{parse_result.scheme}://{parse_result.netloc}'

        if not databricks_token:
            databricks_token = os.environ['DATABRICKS_TOKEN']
        self.databricks_token = databricks_token

        if len(full_name.split('.')) != 3:
            raise ValueError(f'Invalid UC Volume name. Volume should be of the format: <catalog>.<schema>.<volume>')

        self.full_volume_name = full_name
        self.catalog, self.schema, self.volume = self.full_volume_name.split('.')
        log.info(
            f'Initialized UCVolumeObjectStore with full_name={self.full_volume_name} and host={self.databricks_host_name}'
        )

    def get_temporary_credentials(self, operation: Operation) -> S3Credentials:
        from databricks.sdk import WorkspaceClient

        ws_client = WorkspaceClient()  # auth automatically picked up from env variables
        volume_info = ws_client.volumes.read(self.full_volume_name)

        log.info(f'Fetched volume_info={volume_info}')

        storage_location = volume_info.storage_location
        parse_result = urlparse(storage_location)
        backend, bucket, prefix = parse_result.scheme, parse_result.netloc, parse_result.path

        if backend != 's3':
            raise ValueError(f'The remote backend {backend} is not supported for UCVolumes')

        # fetch temporary creds
        url = f'{self.databricks_host_name}/api/2.1/unity-catalog/temporary-volume-credentials'
        data = {'volume_id': volume_info.volume_id, 'operation': operation.name}

        resp = requests.post(url, json=data, headers={'Authorization': f'Bearer {self.databricks_token}'})
        if resp.status_code != 200:
            raise Exception(f'Calling {url} resulted in status_code={resp.status_code} with message {resp.raw}')

        aws_credentials = resp.json()['aws_temp_credentials']

        return S3Credentials(
            bucket=bucket,
            prefix=prefix,
            aws_access_key_id=aws_credentials['access_key_id'],
            aws_session_token=aws_credentials['session_token'],
            aws_secret_access_key=aws_credentials['secret_access_key'],
        )

    def upload_object(self,
                      object_name: str,
                      filename: str | pathlib.Path,
                      callback: Callable[[int, int], None] | None = None) -> None:
        s3_creds = self.get_temporary_credentials(Operation.WRITE_VOLUME)
        log.info(f'Fetched temporary credentials for s3 bucket={s3_creds.bucket} and prefix={s3_creds.prefix}')
        s3_store = S3ObjectStore(
            bucket=s3_creds.bucket,
            prefix=s3_creds.prefix,
            aws_access_key_id=s3_creds.aws_access_key_id,
            aws_session_token=s3_creds.aws_session_token,
            aws_secret_access_key=s3_creds.aws_secret_access_key,
        )
        return s3_store.upload_object(object_name=object_name, filename=filename, callback=callback)

    def download_object(self,
                        object_name: str,
                        filename: str | pathlib.Path,
                        overwrite: bool = False,
                        callback: Callable[[int, int], None] | None = None) -> None:
        s3_creds = self.get_temporary_credentials(Operation.READ_VOLUME)
        s3_store = S3ObjectStore(
            bucket=s3_creds.bucket,
            prefix=s3_creds.prefix,
            aws_access_key_id=s3_creds.aws_access_key_id,
            aws_session_token=s3_creds.aws_session_token,
            aws_secret_access_key=s3_creds.aws_secret_access_key,
        )
        return s3_store.download_object(object_name=object_name,
                                        filename=filename,
                                        overwrite=overwrite,
                                        callback=callback)

    def get_object_size(self, object_name: str) -> int:
        s3_creds = self.get_temporary_credentials(Operation.READ_VOLUME)
        s3_store = S3ObjectStore(
            bucket=s3_creds.bucket,
            prefix=s3_creds.prefix,
            aws_access_key_id=s3_creds.aws_access_key_id,
            aws_session_token=s3_creds.aws_session_token,
            aws_secret_access_key=s3_creds.aws_secret_access_key,
        )
        return s3_store.get_object_size(object_name=object_name)
