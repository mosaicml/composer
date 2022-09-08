# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import pathlib
from typing import Any, Dict, Type

import pytest

import composer
import composer.utils.object_store.object_store_hparams
from composer.utils.object_store import ObjectStore
from composer.utils.object_store.object_store_hparams import (LibcloudObjectStoreHparams, ObjectStoreHparams,
                                                              S3ObjectStoreHparams, SFTPObjectStoreHparams,
                                                              object_store_registry)
from tests.common import get_module_subclasses
from tests.hparams.common import assert_in_registry, construct_from_yaml
from tests.utils.object_store.object_store_settings import get_object_store_ctx

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
    _SFTP_AVAILABLE = True
    del paramiko
except ImportError:
    _SFTP_AVAILABLE = False

object_store_hparam_kwargs: Dict[Type[ObjectStoreHparams], Dict[str, Any]] = {
    S3ObjectStoreHparams: {
        'bucket': 'my-bucket',
    },
    LibcloudObjectStoreHparams: {
        'provider': 'local',
        'key_environ': 'OBJECT_STORE_KEY',
        'container': '.',
    },
    SFTPObjectStoreHparams: {
        'host': 'localhost',
        'port': 23,
        'username': 'test_user',
    }
}

_object_store_marks = {
    LibcloudObjectStoreHparams: [pytest.mark.skipif(not _LIBCLOUD_AVAILABLE, reason='Missing dependency')],
    S3ObjectStoreHparams: [pytest.mark.skipif(not _BOTO3_AVAILABLE, reason='Missing dependency')],
    SFTPObjectStoreHparams: [
        pytest.mark.skipif(not _SFTP_AVAILABLE, reason='Missing dependency'),
        pytest.mark.filterwarnings(r'ignore:setDaemon\(\) is deprecated:DeprecationWarning'),
    ],
}

object_store_hparams = [
    pytest.param(x, marks=_object_store_marks[x], id=x.__name__) for x in get_module_subclasses(
        composer.utils.object_store.object_store_hparams,
        ObjectStoreHparams,
    )
]


@pytest.mark.parametrize('constructor', object_store_hparams)
def test_object_store_hparams_is_constructable(
    constructor: Type[ObjectStoreHparams],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
):
    yaml_dict = object_store_hparam_kwargs[constructor]
    instance = construct_from_yaml(constructor, yaml_dict=yaml_dict)
    with get_object_store_ctx(instance.get_object_store_cls(), yaml_dict, monkeypatch, tmp_path):
        with instance.initialize_object() as object_store:
            assert isinstance(object_store, ObjectStore)


@pytest.mark.parametrize('constructor', object_store_hparams)
def test_hparams_in_registry(constructor: Type[ObjectStoreHparams]):
    assert_in_registry(constructor, object_store_registry)
