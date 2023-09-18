from __future__ import annotations

import logging
import os
import pathlib
import uuid
from typing import Callable

from composer.utils.import_helpers import MissingConditionalImportError
from composer.utils.object_store.object_store import ObjectStore

log = logging.getLogger(__name__)


class UCObjectStore(ObjectStore):
    """Utility class for uploading and downloading data from Databricks Unity Catalog Volumes

    Args:
        uri (str): The Databricks UC Volume URI that is of the format
            `dbfs:/Volumes/<catalog-name>/<schema-name>/<volume-name>/path`

    """

    def __init__(self, uri: str) -> None:
        try:
            import databricks
        except ImportError as e:
            raise MissingConditionalImportError('databricks') from e

        if not 'DATABRICKS_HOST' in os.environ or not 'DATABRICKS_TOKEN' in os.environ:
            raise ValueError('Environment variables `DATABRICKS_HOST` and `DATABRICKS_TOKEN` '
                             'must be set to use Databricks Unity Catalog Volumes')

        if not uri.startswith('dbfs:/Volumes'):
            raise ValueError('Databricks Unity Catalog Volumes paths should start with "dbfs:/Volumes".')
        self.path = uri.lstrip('dbfs:')

        from databricks.sdk import WorkspaceClient
        self.client = WorkspaceClient()

    def get_uri(self, object_name: str) -> str:
        return f'dbfs:{self.get_object_path(object_name)}'

    def get_object_path(self, object_name: str) -> str:
        return os.path.join(self.path, object_name)

    def upload_object(self,
                      object_name: str,
                      filename: str | pathlib.Path,
                      callback: Callable[[int, int], None] | None = None) -> None:
        # remove unused variable
        del callback

        with open(filename, 'rb') as f:
            self.client.files.upload(self.get_object_path(object_name), f)

    def download_object(self,
                        object_name: str,
                        filename: str | pathlib.Path,
                        overwrite: bool = False,
                        callback: Callable[[int, int], None] | None = None) -> None:
        # remove unused variable
        del callback

        if os.path.exists(filename) and not overwrite:
            raise FileExistsError(f'The file at {filename} already exists and overwrite is set to False.')

        dirname = os.path.dirname(filename)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        tmp_path = str(filename) + f'{uuid.uuid4()}.tmp'

        try:
            from databricks.sdk.core import DatabricksError
            try:
                resp = self.client.files.download(self.get_object_path(object_name))
                with open(tmp_path, 'wb') as f:
                    f.write(resp.contents.read())
            except DatabricksError as e:
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
        file_info = self.client.files.get_status(self.get_object_path(object_name))
        return file_info.file_size
