from __future__ import annotations

import logging
import os
import pathlib
import uuid
from typing import Callable

from composer.utils.import_helpers import MissingConditionalImportError
from composer.utils.object_store.object_store import ObjectStore

log = logging.getLogger(__name__)


class DatabricksUnityCatalogVolume(ObjectStore):
    """
    Utility for uploading and downloading data from Databricks Unity Catalog Volumes
    """

    def __init__(self, uri: str) -> None:
        try:
            import databricks
        except ImportError as e:
            raise MissingConditionalImportError('databricks') from e

        prefix = uri.lstrip('uc:/')
        self.prefix = prefix.strip('/')
        self.prefix += '/'

        if not 'DATABRICKS_HOST' not in os.environ or 'DATABRICKS_TOKEN' not in os.environ:
            # TODO: Raise a better exception here
            raise ValueError('Environment variables `DATABRICKS_HOST` and `DATABRICKS_TOKEN` '
                             'must be set to use Databricks Unity Catalog Volumes')

        from databricks.sdk import WorkspaceClient
        self.client = WorkspaceClient()

    def get_uri(self, object_name: str) -> str:
        return f'uc:/{self.get_object_path(object_name)}'

    def get_object_path(self, object_name: str) -> str:
        return os.path.join(self.prefix, object_name)

    # TODO: Figure out if / how we can use callbacks here
    def upload_object(self,
                      object_name: str,
                      filename: str | pathlib.Path,
                      callback: Callable[[int, int], None] | None = None) -> None:
        with open(filename, 'rb') as f:
            self.client.files.upload(self.get_object_path(object_name), f)

    # TODO: Figure out if / how we can use callbacks here
    def download_object(self,
                        object_name: str,
                        filename: str | pathlib.Path,
                        overwrite: bool = False,
                        callback: Callable[[int, int], None] | None = None) -> None:
        if os.path.exists(filename) and not overwrite:
            raise FileExistsError(f'The file at {filename} already exists and overwrite is set to False.')

        dirname = os.path.dirname(filename)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        tmp_path = str(filename) + f'{uuid.uuid4()}.tmp'

        try:
            try:
                uc_file = self.client.files.download(self.get_object_path(object_name))
                with open(tmp_path, 'wb') as f:
                    f.write(uc_file.read())
                uc_file.close()
            except Exception as e:
                raise e
        except:
            # Make best effort attempt to clean up the temporary file
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

    def get_object_size(self, object_name: str) -> int:
        pass

    def get_uri(self, object_name: str) -> str:
        return self.path
