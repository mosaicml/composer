# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import datetime
import os
import pathlib
from unittest.mock import MagicMock, patch

import pytest
import pytest_httpserver

from composer import loggers
from composer.core.time import Time, Timestamp, TimeUnit
from composer.utils import file_helpers
from composer.utils.file_helpers import (ensure_folder_has_no_conflicting_files, ensure_folder_is_empty,
                                         format_name_with_dist, format_name_with_dist_and_time, get_file, is_tar,
                                         maybe_create_object_store_from_uri,
                                         maybe_create_remote_uploader_downloader_from_uri, parse_uri)
from composer.utils.object_store.libcloud_object_store import LibcloudObjectStore
from tests.common.markers import world_size
from tests.loggers.test_remote_uploader_downloader import DummyObjectStore


@pytest.mark.xfail(reason='Occasionally hits the timeout. Should refactor to use a local webserver.')
def test_get_file_uri(tmp_path: pathlib.Path, httpserver: pytest_httpserver.HTTPServer):
    httpserver.expect_request('/hi').respond_with_data('hi')
    get_file(
        path=httpserver.url_for('/hi'),
        object_store=None,
        destination=str(tmp_path / 'example'),
    )
    with open(str(tmp_path / 'example'), 'r') as f:
        assert f.readline().startswith('<!')


@pytest.mark.xfail(reason='Occasionally hits the timeout. Should refactor to use a local webserver.')
def test_get_file_uri_not_found(tmp_path: pathlib.Path, httpserver: pytest_httpserver.HTTPServer):
    with pytest.raises(FileNotFoundError):
        get_file(
            path=httpserver.url_for('/not_found_url'),
            object_store=None,
            destination=str(tmp_path / 'example'),
        )


def test_get_file_object_store(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch):
    pytest.importorskip('libcloud')

    remote_dir = tmp_path / 'remote_dir'
    os.makedirs(remote_dir)
    monkeypatch.setenv('OBJECT_STORE_KEY', str(remote_dir))  # for the local option, the key is the path
    provider = LibcloudObjectStore(
        provider='local',
        key_environ='OBJECT_STORE_KEY',
        container='.',
    )

    with open(str(remote_dir / 'checkpoint.txt'), 'wb') as f:
        f.write(b'checkpoint1')
    get_file(
        path='checkpoint.txt',
        object_store=provider,
        destination=str(tmp_path / 'example'),
    )
    with open(str(tmp_path / 'example'), 'rb') as f:
        assert f.read() == b'checkpoint1'


def test_get_file_auto_object_store(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch):
    with patch('composer.utils.file_helpers.S3ObjectStore', DummyObjectStore):
        object_store = DummyObjectStore(pathlib.Path('my-test-bucket'))
        with open(str(tmp_path / 'test-file.txt'), 'w') as _txt_file:
            _txt_file.write('testing')
        object_store.upload_object('test-file.txt', str(tmp_path / 'test-file.txt'))
        get_file(f's3://my-test-bucket/test-file.txt', str(tmp_path / 'loaded-test-file.txt'))

    with open(str(tmp_path / 'loaded-test-file.txt')) as _txt_file:
        loaded_content = _txt_file.read()

    assert loaded_content.startswith('testing')


def test_get_file_object_store_with_symlink(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch):
    pytest.importorskip('libcloud')

    remote_dir = tmp_path / 'remote_dir'
    os.makedirs(remote_dir)
    monkeypatch.setenv('OBJECT_STORE_KEY', str(remote_dir))  # for the local option, the key is the path
    provider = LibcloudObjectStore(
        provider='local',
        key_environ='OBJECT_STORE_KEY',
        container='.',
    )
    # Add file to object store
    with open(str(remote_dir / 'checkpoint.txt'), 'wb') as f:
        f.write(b'checkpoint1')
    # Add symlink to object store
    with open(str(remote_dir / 'latest.symlink'), 'w') as f:
        f.write('checkpoint.txt')
    # Fetch object, should automatically follow symlink
    get_file(
        path='latest.symlink',
        object_store=provider,
        destination=str(tmp_path / 'example'),
    )
    with open(str(tmp_path / 'example'), 'rb') as f:
        assert f.read() == b'checkpoint1'
    # Fetch object without specifying .symlink, should automatically follow
    get_file(
        path='latest',
        object_store=provider,
        destination=str(tmp_path / 'example'),
        overwrite=True,
    )
    with open(str(tmp_path / 'example'), 'rb') as f:
        assert f.read() == b'checkpoint1'


def test_get_file_object_store_not_found(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch):
    pytest.importorskip('libcloud')

    remote_dir = tmp_path / 'remote_dir'
    os.makedirs(remote_dir)
    monkeypatch.setenv('OBJECT_STORE_KEY', str(remote_dir))  # for the local option, the key is the path
    provider = LibcloudObjectStore(
        provider='local',
        key_environ='OBJECT_STORE_KEY',
        container='.',
    )
    with pytest.raises(FileNotFoundError):
        get_file(
            path='checkpoint.txt',
            object_store=provider,
            destination=str(tmp_path / 'example'),
        )


def test_get_file_local_path(tmp_path: pathlib.Path):
    tmpfile_name = os.path.join(tmp_path, 'file.txt')
    with open(tmpfile_name, 'x') as f:
        f.write('hi!')

    get_file(
        path=tmpfile_name,
        object_store=None,
        destination=str(tmp_path / 'example'),
    )
    with open(str(tmp_path / 'example'), 'r') as f:
        assert f.read() == 'hi!'


def test_get_file_local_path_overwrite_false(tmp_path: pathlib.Path):
    tmpfile_name = os.path.join(tmp_path, 'file.txt')
    with open(tmpfile_name, 'x') as f:
        f.write('hi!')

    with open(str(tmp_path / 'example'), 'w') as f:
        f.write('already exists!')

    with pytest.raises(FileExistsError):
        get_file(path=tmpfile_name, object_store=None, destination=str(tmp_path / 'example'), overwrite=False)
        with open(str(tmp_path / 'example'), 'r') as f:
            assert f.read() == 'hi!'


def test_get_file_local_path_overwrite_true(tmp_path: pathlib.Path):
    tmpfile_name = os.path.join(tmp_path, 'file.txt')
    with open(tmpfile_name, 'x') as f:
        f.write('hi!')

    with open(str(tmp_path / 'example'), 'w') as f:
        f.write('already exists!')

    get_file(path=tmpfile_name, object_store=None, destination=str(tmp_path / 'example'), overwrite=True)
    with open(str(tmp_path / 'example'), 'r') as f:
        assert f.read() == 'hi!'


def test_get_file_local_path_not_found():
    with pytest.raises(FileNotFoundError):
        get_file(
            path='/path/does/not/exist',
            object_store=None,
            destination='destination',
        )


def test_is_tar():
    assert is_tar('x.tar')
    assert is_tar('x.tgz')
    assert is_tar('x.tar.gz')
    assert is_tar('x.tar.bz2')
    assert is_tar('x.tar.lzma')
    assert not is_tar('x')


def test_format_name_with_dist():
    vars = ['run_name', 'rank', 'node_rank', 'world_size', 'local_world_size', 'local_rank', 'extra']
    format_str = ','.join(f'{x}={{{x}}}' for x in vars)
    expected_str = 'run_name=awesome_run,rank=0,node_rank=0,world_size=1,local_world_size=1,local_rank=0,extra=42'
    assert format_name_with_dist(format_str, 'awesome_run', extra=42) == expected_str


@world_size(2)
def test_safe_format_name_with_dist(monkeypatch: pytest.MonkeyPatch, world_size):
    """node rank deleted, but not in format string, so format should complete."""
    vars = ['run_name', 'world_size']
    format_str = ','.join(f'{x}={{{x}}}' for x in vars)
    expected_str = 'run_name=awesome_run,world_size=2'

    monkeypatch.delenv('NODE_RANK')
    assert format_name_with_dist(format_str, 'awesome_run') == expected_str


@world_size(2)
def test_unsafe_format_name_with_dist(monkeypatch: pytest.MonkeyPatch, world_size):
    """Node rank is deleted, but also in the format string, so expect error."""
    vars = ['run_name', 'node_rank']
    format_str = ','.join(f'{x}={{{x}}}' for x in vars)

    monkeypatch.delenv('NODE_RANK')
    with pytest.raises(KeyError):
        assert format_name_with_dist(format_str, 'awesome_run') == 'run_name=awesome_run,node_rank=3'


def test_format_name_with_dist_and_time():
    vars = [
        'run_name',
        'rank',
        'node_rank',
        'world_size',
        'local_world_size',
        'local_rank',
        'extra',
        'epoch',
        'batch',
        'batch_in_epoch',
        'sample',
        'sample_in_epoch',
        'token',
        'token_in_epoch',
        'total_wct',
        'epoch_wct',
        'batch_wct',
    ]
    format_str = ','.join(f'{x}={{{x}}}' for x in vars)
    expected_str = ('run_name=awesome_run,rank=0,node_rank=0,world_size=1,local_world_size=1,local_rank=0,extra=42,'
                    'epoch=0,batch=1,batch_in_epoch=1,sample=2,sample_in_epoch=2,token=3,token_in_epoch=3,'
                    'total_wct=36000.0,epoch_wct=3000.0,batch_wct=5.0')
    timestamp = Timestamp(
        epoch=Time.from_timestring('0ep'),
        batch=Time.from_timestring('1ba'),
        batch_in_epoch=Time.from_timestring('1ba'),
        sample=Time.from_timestring('2sp'),
        sample_in_epoch=Time.from_timestring('2sp'),
        token=Time.from_timestring('3tok'),
        token_in_epoch=Time.from_timestring('3tok'),
        total_wct=datetime.timedelta(hours=10),  # formatted as seconds
        epoch_wct=datetime.timedelta(minutes=50),  # formatted as seconds
        batch_wct=datetime.timedelta(seconds=5),  # formatted as seconds
    )
    assert format_name_with_dist_and_time(format_str, 'awesome_run', timestamp=timestamp, extra=42) == expected_str


@pytest.mark.parametrize('input_uri,expected_parsed_uri', [
    ('backend://bucket/path', ('backend', 'bucket', 'path')),
    ('backend://bucket@namespace/path', ('backend', 'bucket', 'path')),
    ('backend://bucket/a/longer/path', ('backend', 'bucket', 'a/longer/path')),
    ('a/long/path', ('', '', 'a/long/path')),
    ('/a/long/path', ('', '', '/a/long/path')),
    ('backend://bucket/', ('backend', 'bucket', '')),
    ('backend://bucket', ('backend', 'bucket', '')),
    ('backend://', ('backend', '', '')),
])
def test_parse_uri(input_uri, expected_parsed_uri):
    actual_parsed_uri = parse_uri(input_uri)
    assert actual_parsed_uri == expected_parsed_uri


def test_maybe_create_object_store_from_uri(monkeypatch):
    mock_s3_obj = MagicMock()
    monkeypatch.setattr(file_helpers, 'S3ObjectStore', mock_s3_obj)
    mock_oci_obj = MagicMock()
    monkeypatch.setattr(file_helpers, 'OCIObjectStore', mock_oci_obj)
    mock_gs_libcloud_obj = MagicMock()
    monkeypatch.setattr(file_helpers, 'LibcloudObjectStore', mock_gs_libcloud_obj)

    assert maybe_create_object_store_from_uri('checkpoint/for/my/model.pt') is None

    maybe_create_object_store_from_uri('s3://my-bucket/path')
    mock_s3_obj.assert_called_once_with(bucket='my-bucket')

    with pytest.raises(NotImplementedError):
        maybe_create_object_store_from_uri('wandb://my-cool/checkpoint/for/my/model.pt')

    with pytest.raises(ValueError):
        maybe_create_object_store_from_uri('gs://my-bucket/path')

    os.environ['GCS_KEY'] = 'foo'
    os.environ['GCS_SECRET'] = 'foo'
    maybe_create_object_store_from_uri('gs://my-bucket/path')
    mock_gs_libcloud_obj.assert_called_once_with(
        provider='google_storage',
        container='my-bucket',
        key_environ='GCS_KEY',
        secret_environ='GCS_SECRET',
    )
    del os.environ['GCS_KEY']
    del os.environ['GCS_SECRET']

    maybe_create_object_store_from_uri('oci://my-bucket/path')
    mock_oci_obj.assert_called_once_with(bucket='my-bucket')

    with pytest.raises(NotImplementedError):
        maybe_create_object_store_from_uri('ms://bucket/checkpoint/for/my/model.pt')


def test_maybe_create_remote_uploader_downloader_from_uri(monkeypatch):
    assert maybe_create_remote_uploader_downloader_from_uri('checkpoint/for/my/model.pt', loggers=[]) is None
    from composer.loggers import RemoteUploaderDownloader
    mock_remote_ud_obj = MagicMock()
    mock_remote_ud_obj.remote_backend_name = 's3'
    mock_remote_ud_obj.remote_bucket_name = 'my-nifty-bucket'
    mock_remote_ud_obj.__class__ = RemoteUploaderDownloader

    with pytest.warns(Warning, match='There already exists a RemoteUploaderDownloader object to handle'):
        maybe_create_remote_uploader_downloader_from_uri('s3://my-nifty-bucket/path', loggers=[mock_remote_ud_obj])
    del RemoteUploaderDownloader
    with monkeypatch.context() as m:
        mock_remote_ud = MagicMock()
        m.setattr(loggers, 'RemoteUploaderDownloader', mock_remote_ud)
        maybe_create_remote_uploader_downloader_from_uri('s3://my-nifty-s3-bucket/path/to/checkpoints.pt', loggers=[])
        mock_remote_ud.assert_called_once_with(bucket_uri='s3://my-nifty-s3-bucket')

    with monkeypatch.context() as m:
        mock_remote_ud = MagicMock()
        m.setattr(loggers, 'RemoteUploaderDownloader', mock_remote_ud)
        maybe_create_remote_uploader_downloader_from_uri('oci://my-nifty-oci-bucket/path/to/checkpoints.pt', loggers=[])
        mock_remote_ud.assert_called_once_with(bucket_uri='oci://my-nifty-oci-bucket')

    with monkeypatch.context() as m:
        mock_remote_ud = MagicMock()
        m.setattr(loggers, 'RemoteUploaderDownloader', mock_remote_ud)

        with pytest.raises(ValueError):
            maybe_create_remote_uploader_downloader_from_uri('gs://my-nifty-gs-bucket/path/to/checkpoints.pt',
                                                             loggers=[])

        os.environ['GCS_KEY'] = 'foo'
        os.environ['GCS_SECRET'] = 'foo'
        maybe_create_remote_uploader_downloader_from_uri('gs://my-nifty-gs-bucket/path/to/checkpoints.pt', loggers=[])
        mock_remote_ud.assert_called_once_with(bucket_uri='libcloud://my-nifty-gs-bucket',
                                               backend_kwargs={
                                                   'provider': 'google_storage',
                                                   'container': 'my-nifty-gs-bucket',
                                                   'key_environ': 'GCS_KEY',
                                                   'secret_environ': 'GCS_SECRET',
                                               })
        del os.environ['GCS_KEY']
        del os.environ['GCS_SECRET']

    with pytest.raises(NotImplementedError):
        maybe_create_remote_uploader_downloader_from_uri('wandb://my-cool/checkpoint/for/my/model.pt', loggers=[])

    with pytest.raises(NotImplementedError):
        maybe_create_remote_uploader_downloader_from_uri('ms://bucket/checkpoint/for/my/model.pt', loggers=[])


def test_ensure_folder_is_empty(tmp_path: pathlib.Path):
    ensure_folder_is_empty(tmp_path)


@pytest.mark.parametrize(
    'filename,new_file,success',
    [
        [
            'blazing-unicorn-ep{epoch}-batch{batch}-tie{token_in_epoch}-rank{rank}.pt',
            'blazing-unicorn-ep1-batch3-tie6-rank0.pt', True
        ],  # Ignore timestamps in past
        [
            'blazing-unicorn-ep{epoch}-batch{batch}-tie{token_in_epoch}-rank{rank}.pt',
            'blazing-unicorn-ep2-batch6-tie7-rank0.pt', True
        ],  # Ignore timestamps in with same time as current
        [
            'blazing-unicorn-ep{epoch}-batch{batch}-tie{token_in_epoch}-rank{rank}.pt',
            'blazing-unicorn-ep1-batch6-tie9-rank0.pt', True
        ],  # Ignore timestamps with earlier epochs but later samples in epoch
        [
            'blazing-unicorn-ep{epoch}-batch{batch}-tie{token_in_epoch}-rank{rank}.pt',
            'inglorious-monkeys-ep1-batch3-tie6-rank0.pt', True
        ],  # Ignore timestamps of different runs
        [
            'blazing-unicorn-ep{epoch}-batch{batch}-tie{token_in_epoch}-rank{rank}.pt', 'blazing-unicorn-ep3-rank0.pt',
            True
        ],  # Ignore timestamps with same run name but different format
        [
            'blazing-unicorn-ep{epoch}-batch{batch}-tie{token_in_epoch}-rank{rank}.pt',
            'blazing-unicorn-ep3-batch9-tie6-rank0.pt', False
        ],  # Error if in future
        [
            'blazing-unicorn-ep{epoch}-batch{batch}-tie{token_in_epoch}-rank{rank}.pt',
            'blazing-unicorn-ep3-batch9-tie6-rank0.pt', False
        ],  # Error if in future with different rank
        [
            'blazing-unicorn-ep{epoch}-batch{batch}-tie{token_in_epoch}-rank{rank}.pt',
            'blazing-unicorn-ep1-batch9-tie6-rank0.pt', False
        ],  # Error if in future for batches but not epochs
        [
            'blazing-unicorn-ep{epoch}-batch{batch}-tie{token_in_epoch}-rank{rank}.pt',
            'blazing-unicorn-ep2-batch7-tie9-rank0.pt', False
        ],  # Error if in same epoch but later in sample in epoch
        [
            'charging-chungus-ep{epoch}-b{batch}-s{sample}-t{token}-bie{batch_in_epoch}-sie{sample_in_epoch}-tie{token_in_epoch}.pt',
            'charging-chungus-ep1-b3-s6-t12-bie0-sie0-tie0.pt', True
        ],  # Ignore timestamps in past
        [
            'charging-chungus-ep{epoch}-b{batch}-s{sample}-t{token}-bie{batch_in_epoch}-sie{sample_in_epoch}-tie{token_in_epoch}.pt',
            'charging-chungus-ep2-b7-s15-t31-bie1-sie3-tie8.pt', False
        ],  # Error if in future
    ],
)
def test_ensure_folder_has_no_conflicting_files(
    tmp_path: pathlib.Path,
    filename: str,
    new_file: str,
    success: bool,
):
    timestamp = Timestamp(epoch=Time(2, TimeUnit.EPOCH),
                          batch=Time(7, TimeUnit.BATCH),
                          batch_in_epoch=Time(1, TimeUnit.BATCH),
                          sample=Time(15, TimeUnit.SAMPLE),
                          sample_in_epoch=Time(3, TimeUnit.SAMPLE),
                          token=Time(31, TimeUnit.TOKEN),
                          token_in_epoch=Time(7, TimeUnit.TOKEN))

    with open(os.path.join(tmp_path, new_file), 'w') as f:
        f.write('hello')
    if success:
        ensure_folder_has_no_conflicting_files(tmp_path, filename, timestamp)
    else:
        with pytest.raises(FileExistsError):
            ensure_folder_has_no_conflicting_files(tmp_path, filename, timestamp)
