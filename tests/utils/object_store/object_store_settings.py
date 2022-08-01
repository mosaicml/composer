# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import contextlib
import os
import pathlib
from typing import Any, Dict, Type, Union

import mockssh
import moto
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

import composer.utils.object_store
import composer.utils.object_store.object_store_hparams
import composer.utils.object_store.sftp_object_store
from composer.utils.object_store import LibcloudObjectStore, ObjectStore, S3ObjectStore, SFTPObjectStore
from composer.utils.object_store.object_store_hparams import (LibcloudObjectStoreHparams, ObjectStoreHparams,
                                                              S3ObjectStoreHparams, SFTPObjectStoreHparams)
from composer.utils.object_store.sftp_object_store import SFTPObjectStore
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
    _SFTP_AVAILABLE = True
    del paramiko
except ImportError:
    _SFTP_AVAILABLE = False

_object_store_marks = {
    LibcloudObjectStore: [pytest.mark.skipif(not _LIBCLOUD_AVAILABLE, reason='Missing dependency')],
    LibcloudObjectStoreHparams: [pytest.mark.skipif(not _LIBCLOUD_AVAILABLE, reason='Missing dependency')],
    S3ObjectStore: [
        pytest.mark.skipif(not _BOTO3_AVAILABLE, reason='Missing dependency'),
        pytest.mark.filterwarnings(r'ignore::ResourceWarning'),
    ],
    S3ObjectStoreHparams: [pytest.mark.skipif(not _BOTO3_AVAILABLE, reason='Missing dependency')],
    SFTPObjectStore: [
        pytest.mark.skipif(not _SFTP_AVAILABLE, reason='Missing dependency'),
        pytest.mark.filterwarnings(r'ignore:setDaemon\(\) is deprecated:DeprecationWarning'),
    ],
    SFTPObjectStoreHparams: [
        pytest.mark.skipif(not _SFTP_AVAILABLE, reason='Missing dependency'),
        pytest.mark.filterwarnings(r'ignore:setDaemon\(\) is deprecated:DeprecationWarning'),
    ],
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
        'prefix': 'folder/subfolder'
    },
    S3ObjectStoreHparams: {
        'bucket': 'my-bucket',
    },
    LibcloudObjectStoreHparams: {
        'provider': 'local',
        'key_environ': 'OBJECT_STORE_KEY',
        'container': '.',
    },
    SFTPObjectStore: {
        'host': 'localhost',
        'port': 23,
        'username': 'test_user',
    },
    SFTPObjectStoreHparams: {
        'host': 'localhost',
        'port': 23,
        'username': 'test_user',
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


@contextlib.contextmanager
def get_object_store_ctx(object_store_cls: Type[ObjectStore], monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path):
    if object_store_cls is S3ObjectStore:
        pytest.importorskip('boto3')
        import boto3
        monkeypatch.setenv('AWS_ACCESS_KEY_ID', 'testing')
        monkeypatch.setenv('AWS_SECRET_ACCESS_KEY', 'testing')
        monkeypatch.setenv('AWS_SECURITY_TOKEN', 'testing')
        monkeypatch.setenv('AWS_SESSION_TOKEN', 'testing')
        monkeypatch.setenv('AWS_DEFAULT_REGION', 'us-east-1')
        with moto.mock_s3():
            # create the dummy bucket
            s3 = boto3.client('s3')
            s3.create_bucket(Bucket=object_store_kwargs[S3ObjectStore]['bucket'])
            yield
    elif object_store_cls is LibcloudObjectStore:
        pytest.importorskip('libcloud')
        monkeypatch.setenv(object_store_kwargs[LibcloudObjectStoreHparams]['key_environ'], '.')

        remote_dir = tmp_path / 'remote_dir'
        os.makedirs(remote_dir)
        object_store_kwargs[object_store_cls]['provider_kwargs']['key'] = remote_dir
        yield
    elif object_store_cls is SFTPObjectStore:
        pytest.importorskip('paramiko')
        private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        pem = private_key.private_bytes(encoding=serialization.Encoding.PEM,
                                        format=serialization.PrivateFormat.TraditionalOpenSSL,
                                        encryption_algorithm=serialization.NoEncryption())
        private_key_path = tmp_path / 'test_rsa_key'
        username = object_store_kwargs[object_store_cls]['username']
        with open(private_key_path, 'wb') as private_key_file:
            private_key_file.write(pem)
        with mockssh.Server(users={
                username: str(private_key_path),
        }) as server:
            client = server.client(username)
            monkeypatch.setattr(client, 'connect', lambda *args, **kwargs: None)
            monkeypatch.setattr(composer.utils.object_store.sftp_object_store, 'SSHClient', lambda: client)
            yield

    else:
        raise NotImplementedError('Parameterization not implemented')
