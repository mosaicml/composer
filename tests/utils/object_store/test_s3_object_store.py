# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import os
import pathlib
import threading
from unittest import mock
from unittest.mock import ANY, MagicMock

import pytest

from composer.utils.object_store import S3ObjectStore


def _worker(bucket: str, tmp_path: pathlib.Path, tid: int):
    object_store = S3ObjectStore(bucket=bucket)
    os.makedirs(tmp_path / str(tid))
    with pytest.raises(FileNotFoundError):
        object_store.download_object('this_key_should_not_exist', filename=tmp_path / str(tid) / 'dummy_file')


# This test requires properly configured aws credentials; otherwise the s3 client would hit a NoCredentialsError
# when constructing the Session, which occurs before the bug this test checks
@pytest.mark.remote
def test_s3_object_store_multi_threads(tmp_path: pathlib.Path, s3_bucket: str):
    """Test to verify that we do not hit https://github.com/boto/boto3/issues/1592."""
    pytest.importorskip('boto3')
    threads = []
    # Manually tried fewer threads; it seems that 100 is needed to reliably re-produce the bug
    for i in range(100):
        t = threading.Thread(target=_worker, kwargs={'bucket': s3_bucket, 'tid': i, 'tmp_path': tmp_path})
        t.start()
        threads.append(t)
    for t in threads:
        t.join()


def test_s3_upload_object_arguments(tmp_path: pathlib.Path, s3_bucket: str):
    filename = tmp_path / 'localfile.txt'
    filename.touch()
    remote_obj_name = 'remote.txt'

    object_store = S3ObjectStore(bucket=s3_bucket)
    object_store.client.upload_file = MagicMock()

    with mock.patch.dict('os.environ'):
        os.environ.pop('S3_CANNED_ACL', None)

        object_store.upload_object(object_name=remote_obj_name, filename=filename)
        object_store.client.upload_file.assert_called_with(Bucket='my-bucket',
                                                           Key=remote_obj_name,
                                                           Filename=filename,
                                                           Callback=None,
                                                           Config=ANY)

        object_store.upload_object(object_name=remote_obj_name,
                                   filename=filename,
                                   ExtraArgs={'ACL': 'authenticated-read'})
        object_store.client.upload_file.assert_called_with(Bucket='my-bucket',
                                                           Key=remote_obj_name,
                                                           Filename=filename,
                                                           Callback=None,
                                                           Config=ANY,
                                                           ExtraArgs={'ACL': 'authenticated-read'})

        os.environ['S3_CANNED_ACL'] = 'bucket-owner-full-control'

        object_store.upload_object(object_name=remote_obj_name, filename=filename)
        object_store.client.upload_file.assert_called_with(Bucket='my-bucket',
                                                           Key=remote_obj_name,
                                                           Filename=filename,
                                                           Callback=None,
                                                           Config=ANY,
                                                           ExtraArgs={'ACL': 'bucket-owner-full-control'})

        object_store.upload_object(object_name=remote_obj_name, filename=filename, ExtraArgs={'Metadata': {}})
        object_store.client.upload_file.assert_called_with(Bucket='my-bucket',
                                                           Key=remote_obj_name,
                                                           Filename=filename,
                                                           Callback=None,
                                                           Config=ANY,
                                                           ExtraArgs={
                                                               'ACL': 'bucket-owner-full-control',
                                                               'Metadata': {}
                                                           })

        object_store.upload_object(object_name=remote_obj_name,
                                   filename=filename,
                                   ExtraArgs={'ACL': 'authenticated-read'})
        object_store.client.upload_file.assert_called_with(Bucket='my-bucket',
                                                           Key=remote_obj_name,
                                                           Filename=filename,
                                                           Callback=None,
                                                           Config=ANY,
                                                           ExtraArgs={'ACL': 'authenticated-read'})
