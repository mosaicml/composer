# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import contextlib
import os
import pathlib
from typing import Any, Dict, Type

import mockssh
import moto
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

import composer.utils.remote_filesystem
import composer.utils.remote_filesystem.remote_filesystem_hparams
import composer.utils.remote_filesystem.sftp_remote_filesystem
from composer.utils.remote_filesystem import (LibcloudRemoteFilesystem, RemoteFilesystem, S3RemoteFilesystem,
                                              SFTPRemoteFilesystem)
from composer.utils.remote_filesystem.sftp_remote_filesystem import SFTPRemoteFilesystem
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

_remote_filesystem_marks = {
    LibcloudRemoteFilesystem: [pytest.mark.skipif(not _LIBCLOUD_AVAILABLE, reason='Missing dependency')],
    S3RemoteFilesystem: [
        pytest.mark.skipif(not _BOTO3_AVAILABLE, reason='Missing dependency'),
        pytest.mark.filterwarnings(r'ignore::ResourceWarning'),
    ],
    SFTPRemoteFilesystem: [
        pytest.mark.skipif(not _SFTP_AVAILABLE, reason='Missing dependency'),
        pytest.mark.filterwarnings(r'ignore:setDaemon\(\) is deprecated:DeprecationWarning'),
        pytest.mark.filterwarnings(r'ignore:Unknown .* host key:UserWarning')
    ],
}

remote_filesystems = [
    pytest.param(x, marks=_remote_filesystem_marks[x], id=x.__name__)
    for x in get_module_subclasses(composer.utils.remote_filesystem, RemoteFilesystem)
]


@contextlib.contextmanager
def get_remote_filesystem_ctx(remote_filesystem_cls: Type[RemoteFilesystem],
                              remote_filesystem_kwargs: Dict[str, Any],
                              monkeypatch: pytest.MonkeyPatch,
                              tmp_path: pathlib.Path,
                              remote: bool = False):
    if remote_filesystem_cls is S3RemoteFilesystem:
        pytest.importorskip('boto3')
        import boto3
        if remote:
            yield
        else:
            monkeypatch.setenv('AWS_ACCESS_KEY_ID', 'testing')
            monkeypatch.setenv('AWS_SECRET_ACCESS_KEY', 'testing')
            monkeypatch.setenv('AWS_SECURITY_TOKEN', 'testing')
            monkeypatch.setenv('AWS_SESSION_TOKEN', 'testing')
            monkeypatch.setenv('AWS_DEFAULT_REGION', 'us-east-1')
            with moto.mock_s3():
                # create the dummy bucket
                s3 = boto3.client('s3')
                s3.create_bucket(Bucket=remote_filesystem_kwargs['bucket'])
                yield
    elif remote_filesystem_cls is LibcloudRemoteFilesystem:
        pytest.importorskip('libcloud')
        if remote:
            pytest.skip('Libcloud remote filesystem has no remote tests.')
        monkeypatch.setenv(remote_filesystem_kwargs['key_environ'], '.')

        remote_dir = tmp_path / 'remote_dir'
        os.makedirs(remote_dir)
        if 'provider_kwargs' not in remote_filesystem_kwargs:
            remote_filesystem_kwargs['provider_kwargs'] = {}
        remote_filesystem_kwargs['provider_kwargs']['key'] = remote_dir
        yield
    elif remote_filesystem_cls is SFTPRemoteFilesystem:
        pytest.importorskip('paramiko')
        if remote:
            yield
        else:
            private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
            pem = private_key.private_bytes(encoding=serialization.Encoding.PEM,
                                            format=serialization.PrivateFormat.TraditionalOpenSSL,
                                            encryption_algorithm=serialization.NoEncryption())
            private_key_path = tmp_path / 'test_rsa_key'
            username = remote_filesystem_kwargs['username']
            with open(private_key_path, 'wb') as private_key_file:
                private_key_file.write(pem)
            with mockssh.Server(users={
                    username: str(private_key_path),
            }) as server:
                client = server.client(username)
                monkeypatch.setattr(client, 'connect', lambda *args, **kwargs: None)
                monkeypatch.setattr(composer.utils.remote_filesystem.sftp_remote_filesystem, 'SSHClient',
                                    lambda: client)
                yield

    else:
        raise NotImplementedError('Parameterization not implemented')
