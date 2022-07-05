# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import os
import urllib.parse
from typing import Optional

from composer.utils.object_store.object_store import ObjectStore
from composer.utils.object_store.s3_object_store import S3ObjectStore
from composer.utils.object_store.sftp_object_store import SFTPObjectStore


def get_object_store(remote: str, timeout: Optional[float]) -> ObjectStore:
    """Use the correct download handler to download the file

    Args:
        remote (Optional[str]): Remote path (local filesystem).
        timeout (float): How long to wait for file to download before raising an exception.
    """
    if remote.startswith('s3://'):
        return _get_s3_object_store(remote, timeout)
    elif remote.startswith('sftp://'):
        return _get_sftp_object_store(remote)
    else:
        raise ValueError('unsupported upload scheme')


def _get_s3_object_store(remote: str, timeout: Optional[float]) -> S3ObjectStore:
    if timeout is None:
        raise ValueError('Must specify timeout for s3 bucket')
    obj = urllib.parse.urlparse(remote)
    if obj.scheme != 's3':
        raise ValueError(f"Expected obj.scheme to be 's3', got {obj.scheme} for remote={remote}")
    client_config = {'read_timeout': timeout}
    bucket = obj.netloc
    object_store = S3ObjectStore(bucket=bucket, client_config=client_config)
    return object_store


def _get_sftp_object_store(remote: str) -> SFTPObjectStore:
    url = urllib.parse.urlsplit(remote)
    # Parse URL
    if url.scheme.lower() != 'sftp':
        raise ValueError('If specifying a URI, only the sftp scheme is supported.')
    if not url.hostname:
        raise ValueError('If specifying a URI, the URI must include the hostname.')
    if url.query or url.fragment:
        raise ValueError('Query and fragment parameters are not supported as part of a URI.')
    hostname = url.hostname
    port = url.port
    username = url.username
    password = url.password

    # Get SSH key file if specified
    key_filename = os.environ.get('COMPOSER_SFTP_KEY_FILE', None)
    known_hosts_filename = os.environ.get('COMPOSER_SFTP_KNOWN_HOSTS_FILE', None)

    # Default port
    port = port if port else 22

    object_store = SFTPObjectStore(
        host=hostname,
        port=port,
        username=username,
        password=password,
        known_hosts_filename=known_hosts_filename,
        key_filename=key_filename,
    )
    return object_store
