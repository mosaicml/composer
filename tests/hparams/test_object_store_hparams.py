# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import pathlib
from typing import Any, Dict, Type

import pytest

import composer
import composer.utils.remote_filesystem.remote_filesystem_hparams
from composer.utils.remote_filesystem import RemoteFilesystem
from composer.utils.remote_filesystem.remote_filesystem_hparams import (LibcloudRemoteFilesystemHparams,
                                                                        RemoteFilesystemHparams,
                                                                        S3RemoteFilesystemHparams,
                                                                        SFTPRemoteFilesystemHparams,
                                                                        remote_filesystem_registry)
from tests.common import get_module_subclasses
from tests.hparams.common import assert_in_registry, construct_from_yaml
from tests.utils.remote_filesystem.remote_filesystem_settings import get_remote_filesystem_ctx

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

remote_filesystem_hparam_kwargs: Dict[Type[RemoteFilesystemHparams], Dict[str, Any]] = {
    S3RemoteFilesystemHparams: {
        'bucket': 'my-bucket',
    },
    LibcloudRemoteFilesystemHparams: {
        'provider': 'local',
        'key_environ': 'OBJECT_STORE_KEY',
        'container': '.',
    },
    SFTPRemoteFilesystemHparams: {
        'host': 'localhost',
        'port': 23,
        'username': 'test_user',
    }
}

_remote_filesystem_marks = {
    LibcloudRemoteFilesystemHparams: [pytest.mark.skipif(not _LIBCLOUD_AVAILABLE, reason='Missing dependency')],
    S3RemoteFilesystemHparams: [pytest.mark.skipif(not _BOTO3_AVAILABLE, reason='Missing dependency')],
    SFTPRemoteFilesystemHparams: [
        pytest.mark.skipif(not _SFTP_AVAILABLE, reason='Missing dependency'),
        pytest.mark.filterwarnings(r'ignore:setDaemon\(\) is deprecated:DeprecationWarning'),
    ],
}

remote_filesystem_hparams = [
    pytest.param(x, marks=_remote_filesystem_marks[x], id=x.__name__) for x in get_module_subclasses(
        composer.utils.remote_filesystem.remote_filesystem_hparams,
        RemoteFilesystemHparams,
    )
]


@pytest.mark.parametrize('constructor', remote_filesystem_hparams)
def test_remote_filesystem_hparams_is_constructable(
    constructor: Type[RemoteFilesystemHparams],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
):
    yaml_dict = remote_filesystem_hparam_kwargs[constructor]
    instance = construct_from_yaml(constructor, yaml_dict=yaml_dict)
    with get_remote_filesystem_ctx(instance.get_remote_filesystem_cls(), yaml_dict, monkeypatch, tmp_path):
        with instance.initialize_object() as remote_filesystem:
            assert isinstance(remote_filesystem, RemoteFilesystem)


@pytest.mark.parametrize('constructor', remote_filesystem_hparams)
def test_hparams_in_registry(constructor: Type[RemoteFilesystemHparams]):
    assert_in_registry(constructor, remote_filesystem_registry)
