# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import contextlib
import copy
import pathlib
from typing import Any, Dict, Tuple
from urllib.parse import urlparse

import pytest

from composer.utils.remote_filesystem import (LibcloudRemoteFilesystem, RemoteFilesystem, S3RemoteFilesystem,
                                              SFTPRemoteFilesystem)
from composer.utils.remote_filesystem.sftp_remote_filesystem import SFTPRemoteFilesystem
from tests.utils.remote_filesystem.remote_filesystem_settings import get_remote_filesystem_ctx, remote_filesystems


@pytest.fixture
def bucket_uri_and_kwargs(request, s3_bucket: str, sftp_uri: str, test_session_name: str):
    remote = request.node.get_closest_marker('remote') is not None

    if request.param is LibcloudRemoteFilesystem:
        if remote:
            pytest.skip('Libcloud remote filesystem has no remote tests')
        else:
            bucket_uri = 'libcloud://.'
            kwargs = {
                'provider': 'local',
                'container': '.',
                'key_environ': 'OBJECT_STORE',
                'provider_kwargs': {
                    'key': '.',
                },
            }
    elif request.param is S3RemoteFilesystem:
        if remote:
            bucket_uri = f's3://{s3_bucket}'
            kwargs = {'bucket': s3_bucket, 'prefix': test_session_name}
        else:
            bucket_uri = 's3://my-bucket'
            kwargs = {'bucket': 'my-bucket', 'prefix': 'folder/subfolder'}
    elif request.param is SFTPRemoteFilesystem:
        if remote:
            bucket_uri = f"sftp://{sftp_uri.rstrip('/') + '/' + test_session_name}"
            kwargs = {
                'host': sftp_uri.rstrip('/') + '/' + test_session_name,
                'missing_host_key_policy': 'WarningPolicy',
            }
        else:
            bucket_uri = 'sftp://localhost:23'
            kwargs = {
                'host': 'localhost',
                'port': 23,
                'username': 'test_user',
            }
    else:
        raise ValueError(f'Invalid remote filesystem type: {request.param.__name__}')
    return bucket_uri, kwargs


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


@pytest.mark.parametrize('bucket_uri_and_kwargs', remote_filesystems, indirect=True)
@pytest.mark.parametrize('remote', [False, pytest.param(True, marks=pytest.mark.remote)])
class TestRemoteFilesystem:

    @pytest.fixture
    def remote_filesystem(
        self,
        bucket_uri_and_kwargs: Tuple[str, Dict[str, Any]],
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: pathlib.Path,
        remote: bool,
    ):
        remote_filesystem_name_to_class = {
            's3': S3RemoteFilesystem,
            'sftp': SFTPRemoteFilesystem,
            'libcloud': LibcloudRemoteFilesystem
        }
        bucket_uri, kwargs = bucket_uri_and_kwargs
        remote_filesystem_name = urlparse(bucket_uri).scheme
        with get_remote_filesystem_ctx(remote_filesystem_name_to_class[remote_filesystem_name],
                                       kwargs,
                                       monkeypatch,
                                       tmp_path,
                                       remote=remote):
            copied_config = copy.deepcopy(kwargs)
            # type error: Type[RemoteFilesystem] is not callable
            remote_filesystem = remote_filesystem_name_to_class[remote_filesystem_name](**copied_config)  # type: ignore
            with remote_filesystem:
                yield remote_filesystem

    @pytest.fixture
    def dummy_obj(self, tmp_path: pathlib.Path):
        tmpfile_path = tmp_path / 'file_to_upload'
        with open(tmpfile_path, 'w+') as f:
            f.write('dummy content')
        return tmpfile_path

    def test_upload(self, remote_filesystem: RemoteFilesystem, dummy_obj: pathlib.Path, remote: bool):
        del remote  # unused
        remote_file_name = 'tmpfile_remote_file_name'
        cb = MockCallback(dummy_obj.stat().st_size)
        remote_filesystem.upload_file(remote_file_name, str(dummy_obj), callback=cb)
        cb.assert_all_data_transferred()

    def test_get_uri(self, remote_filesystem: RemoteFilesystem, remote: bool):
        if remote:
            pytest.skip('This test_get_uri does not make any remote calls.')
        uri = remote_filesystem.get_uri('tmpfile_remote_file_name')
        if isinstance(remote_filesystem, S3RemoteFilesystem):
            assert uri == 's3://my-bucket/folder/subfolder/tmpfile_remote_file_name'
        elif isinstance(remote_filesystem, LibcloudRemoteFilesystem):
            assert uri == 'local://./tmpfile_remote_file_name'
        elif isinstance(remote_filesystem, SFTPRemoteFilesystem):
            assert uri == 'sftp://test_user@localhost:23/tmpfile_remote_file_name'
        else:
            raise NotImplementedError(f'Object store {type(remote_filesystem)} not implemented.')

    def test_get_file_size(self, remote_filesystem: RemoteFilesystem, dummy_obj: pathlib.Path, remote: bool):
        del remote  # unused
        remote_file_name = 'tmpfile_remote_file_name'
        remote_filesystem.upload_file(remote_file_name, str(dummy_obj))
        assert remote_filesystem.get_file_size(remote_file_name) == dummy_obj.stat().st_size

    def test_get_file_size_not_found(self, remote_filesystem: RemoteFilesystem, remote: bool):
        del remote  # unused
        with pytest.raises(FileNotFoundError):
            remote_filesystem.get_file_size('not found object')

    @pytest.mark.parametrize('overwrite', [True, False])
    def test_download(
        self,
        remote_filesystem: RemoteFilesystem,
        dummy_obj: pathlib.Path,
        tmp_path: pathlib.Path,
        overwrite: bool,
        remote: bool,
    ):
        del remote  # unused
        remote_file_name = 'tmpfile_remote_file_name'
        remote_filesystem.upload_file(remote_file_name, str(dummy_obj))
        filepath = str(tmp_path / 'destination_path')
        cb = MockCallback(dummy_obj.stat().st_size)
        remote_filesystem.download_file(remote_file_name, filepath, callback=cb)
        ctx = contextlib.nullcontext() if overwrite else pytest.raises(FileExistsError)
        with ctx:
            remote_filesystem.download_file(remote_file_name, filepath, callback=cb, overwrite=overwrite)
        cb.assert_all_data_transferred()

    def test_download_not_found(self, remote_filesystem: RemoteFilesystem, remote: bool):
        with pytest.raises(FileNotFoundError):
            remote_filesystem.download_file('not_found_object', filename='not used')


@pytest.mark.filterwarnings(r'ignore:setDaemon\(\) is deprecated:DeprecationWarning')
def test_filenames_as_environs(monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path):

    key_filepath = tmp_path / 'keyfile'
    key_filepath.touch()

    monkeypatch.setenv('COMPOSER_SFTP_KEY_FILE', str(key_filepath))

    hosts_file = tmp_path / 'host_file'
    hosts_file.touch()

    monkeypatch.setenv('COMPOSER_SFTP_KNOWN_HOSTS_FILE', str(hosts_file))

    kwargs = {
        'host': 'host',
        'username': 'tester',
    }

    with get_remote_filesystem_ctx(SFTPRemoteFilesystem, kwargs, monkeypatch, tmp_path):
        remote_filesystem = SFTPRemoteFilesystem(**kwargs)

        assert remote_filesystem.key_filename == str(key_filepath)
        assert remote_filesystem.known_hosts_filename == str(hosts_file)
