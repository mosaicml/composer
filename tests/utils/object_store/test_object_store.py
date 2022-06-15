# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import contextlib
import os
import pathlib
import tempfile
from typing import Generator, Optional

import mockssh
import moto
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

from composer.utils.object_store import LibcloudObjectStore, ObjectStore, S3ObjectStore, SFTPObjectStore
from composer.utils.object_store.sftp_object_store import SFTPObjectStore
from tests.utils.object_store.object_store_settings import object_store_kwargs


class MockCallback:

    def __init__(self, total_num_bytes: int) -> None:
        self.total_num_bytes = total_num_bytes
        self.transferred_bytes = 0
        self.num_calls = 0

    def __call__(self, transferred: int, total: int):
        self.num_calls += 1
        assert transferred == 0 or transferred >= self.transferred_bytes, 'transferred should be monotonically increasing'
        self.transferred_bytes = transferred
        assert total == self.total_num_bytes

    def assert_all_data_transferred(self):
        assert self.total_num_bytes == self.transferred_bytes


class MockSFTPObjectStore(SFTPObjectStore):

    def __init__(self,
                 host: str,
                 port: int = 22,
                 username: Optional[str] = None,
                 key_file_path: Optional[str] = None,
                 cwd: Optional[str] = None):
        super().__init__(host, port, username, key_file_path, cwd)

    def _create_sftp_client(self):
        server = mockssh.Server(users={})
        private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        pem = private_key.private_bytes(encoding=serialization.Encoding.PEM,
                                        format=serialization.PrivateFormat.TraditionalOpenSSL,
                                        encryption_algorithm=serialization.NoEncryption())
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = os.path.join(tmpdir, 'test_rsa_key')
            private_key_file = open(tmppath, 'wb')
            private_key_file.write(pem)
            private_key_file.close()
            server.add_user(uid=self.username, private_key_path=tmppath)
            server.__enter__()
            self.ssh_client = server.client(self.username)
            self.sftp_client = self.ssh_client.open_sftp()
            return self.sftp_client


@pytest.fixture
def object_store(request, monkeypatch: pytest.MonkeyPatch,
                 tmp_path: pathlib.Path) -> Generator[ObjectStore, None, None]:
    if request.param is S3ObjectStore:
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

            # Yield our object store
            yield request.param(**object_store_kwargs[request.param])
    elif request.param is LibcloudObjectStore:
        pytest.importorskip('libcloud')

        remote_dir = tmp_path / 'remote_dir'
        os.makedirs(remote_dir)
        object_store_kwargs[request.param]['provider_kwargs']['key'] = remote_dir
        yield request.param(**object_store_kwargs[request.param])
    elif request.param is SFTPObjectStore:
        yield MockSFTPObjectStore("test_hostname", port=24, username='test_user')
    else:
        raise NotImplementedError('Parameterization not implemented')


@pytest.mark.parametrize('object_store', [S3ObjectStore, LibcloudObjectStore, SFTPObjectStore], indirect=True)
class TestObjectStore:

    @pytest.fixture
    def dummy_obj(self, object_store: ObjectStore, tmp_path: pathlib.Path):
        tmpfile_path = tmp_path / 'file_to_upload'
        with open(tmpfile_path, 'w+') as f:
            f.write('dummy content')
        return tmpfile_path

    def test_upload(self, object_store: ObjectStore, dummy_obj: pathlib.Path):
        object_name = 'tmpfile_object_name'
        cb = MockCallback(dummy_obj.stat().st_size)
        object_store.upload_object(object_name, str(dummy_obj), callback=cb)
        cb.assert_all_data_transferred()

    def test_get_uri(self, object_store: ObjectStore):
        uri = object_store.get_uri('tmpfile_object_name')
        if isinstance(object_store, S3ObjectStore):
            assert uri == 's3://my-bucket/tmpfile_object_name'
        elif isinstance(object_store, LibcloudObjectStore):
            assert uri == 'local://./tmpfile_object_name'
        elif isinstance(object_store, SFTPObjectStore):
            assert uri == 'sftp://test_hostname:24/tmpfile_object_name'
        else:
            raise NotImplementedError(f'Object store {type(object_store)} not implemented.')

    def test_get_file_size(self, object_store: ObjectStore, dummy_obj: pathlib.Path):
        object_name = 'tmpfile_object_name'
        object_store.upload_object(object_name, str(dummy_obj))
        assert object_store.get_object_size(object_name) == dummy_obj.stat().st_size

    def test_get_file_size_not_found(self, object_store: ObjectStore):
        with pytest.raises(FileNotFoundError):
            object_store.get_object_size('not found object')

    @pytest.mark.parametrize('overwrite', [True, False])
    def test_download(self, object_store: ObjectStore, dummy_obj: pathlib.Path, tmp_path: pathlib.Path,
                      overwrite: bool):
        object_name = 'tmpfile_object_name'
        object_store.upload_object(object_name, str(dummy_obj))
        filepath = str(tmp_path / 'destination_path')
        cb = MockCallback(dummy_obj.stat().st_size)
        object_store.download_object(object_name, filepath, callback=cb)
        ctx = contextlib.nullcontext() if overwrite else pytest.raises(FileExistsError)
        with ctx:
            object_store.download_object(object_name, filepath, callback=cb, overwrite=overwrite)
        cb.assert_all_data_transferred()

    def test_download_not_found(self, object_store: ObjectStore):
        with pytest.raises(FileNotFoundError):
            object_store.download_object('not_found_object', filename='not used')
