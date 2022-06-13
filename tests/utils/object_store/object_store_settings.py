# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict, Type, Union
from composer.utils.object_store.sftp_object_store import SFTPObjectStore

import pytest

import composer.utils.object_store
import composer.utils.object_store.object_store_hparams
from composer.utils.object_store import LibcloudObjectStore, ObjectStore, S3ObjectStore, SFTPObjectStore
from composer.utils.object_store.object_store_hparams import (LibcloudObjectStoreHparams, ObjectStoreHparams,
                                                              S3ObjectStoreHparams, SFTPObjectStoreHparams)
from tests.common import get_module_subclasses

try:
    import libcloud
    _LIBCLOUD_AVAILABLE = True
    del libcloud
except ImportError:
    _LIBCLOUD_AVAILABLE = False

try:
    import boto3
    _BOTO3_AVAILABLE = True
    del boto3
except ImportError:
    _BOTO3_AVAILABLE = False

try:
    import paramiko
    _PARAMIKO_AVAILABLE = True
    del paramiko
except ImportError:
    _PARAMIKO_AVAILABLE = False

_object_store_marks = {
    LibcloudObjectStore: [pytest.mark.skipif(not _LIBCLOUD_AVAILABLE, reason='Missing dependency')],
    LibcloudObjectStoreHparams: [pytest.mark.skipif(not _LIBCLOUD_AVAILABLE, reason='Missing dependency')],
    S3ObjectStore: [pytest.mark.skipif(not _BOTO3_AVAILABLE, reason='Missing dependency')],
    S3ObjectStoreHparams: [pytest.mark.skipif(not _BOTO3_AVAILABLE, reason='Missing dependency')],
    SFTPObjectStore: [pytest.mark.skipif(not _PARAMIKO_AVAILABLE, reason='Missing dependency')],
    SFTPObjectStoreHparams: [pytest.mark.skipif(not _PARAMIKO_AVAILABLE, reason='Missing dependency')],
}

object_store_kwargs: Dict[Union[Type[ObjectStore], Type[ObjectStoreHparams]], Dict[str, Any]] = {
    LibcloudObjectStore: {
        'provider': 'local',
        'container': '.',
        'provider_kwargs': {
            'key': '.',
        },
    },
    S3ObjectStore: {
        'bucket': 'my-bucket',
    },
    S3ObjectStoreHparams: {
        'bucket': 'my-bucket',
    },
    LibcloudObjectStoreHparams: {
        'provider': 'local',
        'key_environ': 'OBJECT_STORE_KEY',
        'container': '.',
    }
}

object_stores = [
    pytest.param(x, marks=_object_store_marks[x], id=x.__name__)
    for x in get_module_subclasses(composer.utils.object_store, ObjectStore)
]
object_store_hparams = [
    pytest.param(x, marks=_object_store_marks[x], id=x.__name__)
    for x in get_module_subclasses(composer.utils.object_store.object_store_hparams, ObjectStoreHparams)
]
