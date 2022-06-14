# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import os
import pathlib

import pytest

from composer.utils.object_store import LibcloudObjectStore


@pytest.fixture
def remote_dir(tmp_path: pathlib.Path):
    remote_dir = tmp_path / 'remote_dir'
    os.makedirs(remote_dir)
    return remote_dir


@pytest.fixture
def local_dir(tmp_path: pathlib.Path):
    local_dir = tmp_path / 'local_dir'
    os.makedirs(local_dir)
    return local_dir


def _get_provider(remote_dir: pathlib.Path, chunk_size: int = 1024 * 1024):
    return LibcloudObjectStore(
        provider='local',
        container='.',
        provider_kwargs={
            'key': str(remote_dir),
        },
        chunk_size=chunk_size,
    )


@pytest.mark.parametrize('chunk_size', [100, 128])
def test_libcloud_object_store_callback(remote_dir: pathlib.Path, local_dir: pathlib.Path, chunk_size: int):
    pytest.importorskip('libcloud')

    provider = _get_provider(remote_dir, chunk_size=chunk_size)
    local_file_path = os.path.join(local_dir, 'dummy_file')
    total_len = 1024
    with open(local_file_path, 'w+') as f:
        f.write('H' * total_len)

    num_calls = 0
    total_bytes_written = 0

    def cb(bytes_written, total_bytes):
        nonlocal num_calls, total_bytes_written
        assert total_bytes == total_len
        num_calls += 1
        total_bytes_written = bytes_written

    provider.upload_object('upload_object', local_file_path, callback=cb)
    # the expected num calls should be 1 more than the ceiling division
    expected_num_calls = (total_len - 1) // chunk_size + 1 + 1
    assert num_calls == expected_num_calls
    assert total_bytes_written == total_len

    num_calls = 0
    total_bytes_written = 0

    local_file_path_download = os.path.join(local_dir, 'dummy_file_downloaded')
    provider.download_object('upload_object', local_file_path_download, callback=cb)

    assert total_bytes_written == total_len
    assert num_calls == expected_num_calls
