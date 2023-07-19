from __future__ import annotations

import logging
import os
import pathlib
import time
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


def timed(func):

    def inner(*args, **kwargs):
        start = time.time()
        func(*args, **kwargs)
        end = time.time()
        total_time_sec = int(end - start)
        log.info(f"Function `{func.__module__}.{func.__name__}(..)` took {total_time_sec} seconds to execute.")

    return inner


class UCVolumeObjectStore(ObjectStore):
    """
    Utility for uploading and downloading models from Unity Catalog Volumes
    """

    def __init__(
        self,
        volume_name: str,
        prefix: str,
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

        if len(volume_name.split('.')) != 3:
            raise ValueError(f'Invalid UC Volume name. Volume should be of the format: <catalog>.<schema>.<volume>')

        self.volume_name = volume_name
        self.catalog, self.schema, self.volume = self.volume_name.split('.')
        log.info(
            f'Initialized UCVolumeObjectStore with full_name={self.volume_name} and host={self.databricks_host_name}')
        self.prefix = prefix.strip('/')
        if self.prefix:
            self.prefix += '/'

    def get_temporary_s3_client(self, operation: Operation) -> S3ObjectStore:
        from databricks.sdk import WorkspaceClient

        ws_client = WorkspaceClient()  # auth automatically picked up from env variables
        volume_info = ws_client.volumes.read(self.volume_name)

        log.info(f'Fetched volume_info={volume_info}')

        storage_location = volume_info.storage_location
        parse_result = urlparse(storage_location)
        backend, bucket, storage_prefix = parse_result.scheme, parse_result.netloc, parse_result.path
        if storage_prefix:
            storage_prefix = storage_prefix.strip('/')
        storage_prefix += '/'

        if backend != 's3':
            raise ValueError(f'The remote backend {backend} is not supported for UCVolumes')

        # fetch temporary creds
        url = f'{self.databricks_host_name}/api/2.1/unity-catalog/temporary-volume-credentials'
        data = {'volume_id': volume_info.volume_id, 'operation': operation.name}

        resp = requests.post(url, json=data, headers={'Authorization': f'Bearer {self.databricks_token}'})
        if resp.status_code != 200:
            raise Exception(f'Calling {url} resulted in status_code={resp.status_code} with message {resp.raw}')

        parsed_resp = resp.json()
        expiration_time_seconds = int(parsed_resp['expiration_time'] / 1000)
        aws_credentials = parsed_resp['aws_temp_credentials']

        epoch_time_now = int(time.time())
        log.info(f'UC credentials expire in {expiration_time_seconds - epoch_time_now} seconds.')

        return S3ObjectStore(
            bucket=bucket,
            prefix=f'{storage_prefix}{self.prefix}',
            aws_access_key_id=aws_credentials['access_key_id'],
            aws_session_token=aws_credentials['session_token'],
            aws_secret_access_key=aws_credentials['secret_access_key'],
        )

    @timed
    def upload_object(self,
                      object_name: str,
                      filename: str | pathlib.Path,
                      callback: Callable[[int, int], None] | None = None) -> None:
        s3_client = self.get_temporary_s3_client(Operation.WRITE_VOLUME)
        log.info(f'Fetched temporary client for s3 bucket={s3_client.bucket} and prefix={s3_client.prefix}')
        return s3_client.upload_object(object_name=object_name, filename=filename, callback=callback)

    def download_object(self,
                        object_name: str,
                        filename: str | pathlib.Path,
                        overwrite: bool = False,
                        callback: Callable[[int, int], None] | None = None) -> None:
        s3_client = self.get_temporary_s3_client(Operation.READ_VOLUME)
        return s3_client.download_object(object_name=object_name,
                                         filename=filename,
                                         overwrite=overwrite,
                                         callback=callback)

    def get_object_size(self, object_name: str) -> int:
        s3_client = self.get_temporary_s3_client(Operation.READ_VOLUME)
        return s3_client.get_object_size(object_name=object_name)
