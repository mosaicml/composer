# Copyright 2024 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Helpers for working with object stores."""

from typing import Any

from composer.utils.object_store.gcs_object_store import GCSObjectStore
from composer.utils.object_store.libcloud_object_store import LibcloudObjectStore
from composer.utils.object_store.mlflow_object_store import MLFLOW_DBFS_PATH_PREFIX, MLFlowObjectStore
from composer.utils.object_store.oci_object_store import OCIObjectStore
from composer.utils.object_store.s3_object_store import S3ObjectStore
from composer.utils.object_store.sftp_object_store import SFTPObjectStore
from composer.utils.object_store.uc_object_store import UCObjectStore

__all__ = ['build_remote_backend']


def build_remote_backend(remote_backend_name: str, backend_kwargs: dict[str, Any]):
    """Build object store given the backend name and kwargs."""
    remote_backend_cls = None
    remote_backend_name_to_cls = {
        's3': S3ObjectStore,
        'oci': OCIObjectStore,
        'sftp': SFTPObjectStore,
        'libcloud': LibcloudObjectStore,
        'gs': GCSObjectStore,
    }

    # Handle `dbfs` backend as a special case, since it can map to either :class:`.UCObjectStore`
    # or :class:`.MLFlowObjectStore`.
    if remote_backend_name == 'dbfs':
        path = backend_kwargs['path']
        if path.startswith(MLFLOW_DBFS_PATH_PREFIX):
            remote_backend_cls = MLFlowObjectStore
        else:
            # Validate if the path conforms to the requirements for UC volume paths
            UCObjectStore.validate_path(path)
            remote_backend_cls = UCObjectStore
    else:
        remote_backend_cls = remote_backend_name_to_cls.get(remote_backend_name, None)
        if remote_backend_cls is None:
            supported_remote_backends = list(remote_backend_name_to_cls.keys()) + ['dbfs']
            raise ValueError(
                f'The remote backend {remote_backend_name} is not supported. Please use one of ({supported_remote_backends})',
            )

    return remote_backend_cls(**backend_kwargs)
