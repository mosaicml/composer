# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import time
from pathlib import Path

import pytest

from composer.utils import GsObjectStore

__DUMMY_OBJ__ = '/tmp/dummy.ckpt'
__NUM_BYTES__ = 1000
bucket_name = 'mosaicml-composer-tests'


@pytest.fixture
def gs_object_store():
    remote_dir = 'gs://mosaicml-composer-tests/streaming/'
    yield GsObjectStore(remote_dir)


# Todo: call trainer to generate a ckpt and resume the train from cloud
def test_train_resuming():
    pass


def test_bucket_not_found():
    with pytest.raises(FileNotFoundError):
        _ = GsObjectStore('gs://not_a_bucket/streaming')


def test_get_uri(gs_object_store):
    object_name = 'test-object'
    expected_uri = 'gs://mosaicml-composer-tests/streaming/test-object'
    assert (gs_object_store.get_uri(object_name) == expected_uri)


def test_get_key(gs_object_store):
    object_name = 'test-object'
    expected_key = 'streaming/test-object'
    assert (gs_object_store.get_key(object_name) == expected_key)


@pytest.mark.parametrize('result', ['success', 'not found'])
def test_get_object_size(gs_object_store, result: str):
    fn = Path(__DUMMY_OBJ__)
    with open(fn, 'wb') as fp:
        fp.write(bytes('0' * __NUM_BYTES__, 'utf-8'))
    gs_object_store.upload_blob(fn)

    if result == 'success':
        assert (gs_object_store.get_object_size(__DUMMY_OBJ__) == __NUM_BYTES__)
    else:  # not found
        with pytest.raises(FileNotFoundError):
            gs_object_store.get_object_size(__DUMMY_OBJ__ + f'time.ctime()')


def test_upload_object(gs_object_store):
    from google.cloud.storage import Blob
    destination_blob_name = '/tmp/dummy.ckpt2'
    key = gs_object_store.get_key(destination_blob_name)
    stats = Blob(bucket=gs_object_store.bucket, name=key).exists(gs_object_store.client)
    if not stats:
        gs_object_store.upload_blob(__DUMMY_OBJ__, destination_blob_name)


@pytest.mark.parametrize('result', ['success', 'file_exists', 'obj_not_found'])
def test_download_object(gs_object_store, tmp_path, result: str):
    fn = Path(__DUMMY_OBJ__)
    with open(fn, 'wb') as fp:
        fp.write(bytes('0' * __NUM_BYTES__, 'utf-8'))
    gs_object_store.upload_blob(fn)

    object_name = __DUMMY_OBJ__
    filename = './dummy.ckpt.download'

    if result == 'success':
        gs_object_store.download_blob(object_name, filename, overwrite=True)

    elif result == 'file_exists':
        with pytest.raises(FileExistsError):
            gs_object_store.download_blob(object_name, __DUMMY_OBJ__)
    else:  # obj_not_found
        with pytest.raises(FileNotFoundError):
            gs_object_store.download_blob(object_name + f'{time.ctime()}', filename, overwrite=True)
