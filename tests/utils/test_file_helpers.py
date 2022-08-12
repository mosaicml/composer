# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import datetime
import os
import pathlib

import pytest
import pytest_httpserver

from composer.core.time import Time, Timestamp, TimeUnit
from composer.utils.file_helpers import (ensure_folder_has_no_conflicting_files, ensure_folder_is_empty,
                                         format_name_with_dist, format_name_with_dist_and_time, get_file, is_tar)
from composer.utils.object_store.object_store_hparams import LibcloudObjectStoreHparams


@pytest.mark.xfail(reason='Occassionally hits the timeout. Should refactor to use a local webserver.')
def test_get_file_uri(tmp_path: pathlib.Path, httpserver: pytest_httpserver.HTTPServer):
    httpserver.expect_request('/hi').respond_with_data('hi')
    get_file(
        path=httpserver.url_for('/hi'),
        object_store=None,
        destination=str(tmp_path / 'example'),
    )
    with open(str(tmp_path / 'example'), 'r') as f:
        assert f.readline().startswith('<!')


@pytest.mark.xfail(reason='Occassionally hits the timeout. Should refactor to use a local webserver.')
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
    provider = LibcloudObjectStoreHparams(
        provider='local',
        key_environ='OBJECT_STORE_KEY',
        container='.',
    ).initialize_object()
    with open(str(remote_dir / 'checkpoint.txt'), 'wb') as f:
        f.write(b'checkpoint1')
    get_file(
        path='checkpoint.txt',
        object_store=provider,
        destination=str(tmp_path / 'example'),
    )
    with open(str(tmp_path / 'example'), 'rb') as f:
        assert f.read() == b'checkpoint1'


def test_get_file_object_store_with_symlink(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch):
    pytest.importorskip('libcloud')

    remote_dir = tmp_path / 'remote_dir'
    os.makedirs(remote_dir)
    monkeypatch.setenv('OBJECT_STORE_KEY', str(remote_dir))  # for the local option, the key is the path
    provider = LibcloudObjectStoreHparams(
        provider='local',
        key_environ='OBJECT_STORE_KEY',
        container='.',
    ).initialize_object()
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
    provider = LibcloudObjectStoreHparams(
        provider='local',
        key_environ='OBJECT_STORE_KEY',
        container='.',
    ).initialize_object()
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
