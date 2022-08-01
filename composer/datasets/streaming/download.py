# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Download handling for :class:`StreamingDataset`.
"""

import gzip
import os
import shutil
import time
import urllib.parse
from typing import Optional

from composer.datasets.streaming.format import split_compression_suffix
from composer.utils import get_file
from composer.utils.object_store.object_store import ObjectStore
from composer.utils.object_store.s3_object_store import S3ObjectStore
from composer.utils.object_store.sftp_object_store import SFTPObjectStore

__all__ = ['download_or_wait', 'get_object_store']


def download_from_http(remote: str, local: str) -> None:
    """Download a file from a http/https remote to local."""
    get_file(path=remote, destination=local, overwrite=True)


def get_object_store(remote: str) -> ObjectStore:
    """Use the correct download handler to download the file

    Args:
        remote (Optional[str]): Remote path (local filesystem).
        timeout (float): How long to wait for file to download before raising an exception.
    """
    if remote.startswith('s3://'):
        return _get_s3_object_store(remote)
    elif remote.startswith('sftp://'):
        return _get_sftp_object_store(remote)
    else:
        raise ValueError('unsupported upload scheme')


def _get_s3_object_store(remote: str) -> S3ObjectStore:
    obj = urllib.parse.urlparse(remote)
    if obj.scheme != 's3':
        raise ValueError(f"Expected obj.scheme to be 's3', got {obj.scheme} for remote={remote}")
    bucket = obj.netloc
    object_store = S3ObjectStore(bucket=bucket)
    return object_store


def _get_sftp_object_store(remote: str) -> SFTPObjectStore:
    # Get SSH key file if specified
    key_filename = os.environ.get('COMPOSER_SFTP_KEY_FILE', None)
    known_hosts_filename = os.environ.get('COMPOSER_SFTP_KNOWN_HOSTS_FILE', None)

    object_store = SFTPObjectStore(
        host=remote,
        known_hosts_filename=known_hosts_filename,
        key_filename=key_filename,
    )
    return object_store


__all__ = ['download_or_wait']


def download_from_local(remote: str, local: str) -> None:
    """Download a file from remote to local.

    Args:
        remote (str): Remote path (local filesystem).
        local (str): Local path (local filesystem).
    """
    local_tmp = local + '.tmp'
    if os.path.exists(local_tmp):
        os.remove(local_tmp)
    shutil.copy(remote, local_tmp)
    os.rename(local_tmp, local)


def dispatch_download(remote: Optional[str], local: str):
    """Use the correct download handler to download the file

    Args:
        remote (Optional[str]): Remote path (local filesystem).
        local (str): Local path (local filesystem).
        timeout (float): How long to wait for file to download before raising an exception.
    """

    local_decompressed, compression_scheme = split_compression_suffix(local)
    if os.path.exists(local_decompressed):
        return

    local_dir = os.path.dirname(local)
    os.makedirs(local_dir, exist_ok=True)

    if not remote:
        raise ValueError('In the absence of local dataset, path to remote dataset must be provided')
    elif remote.startswith('s3://') or remote.startswith('sftp://'):
        url = urllib.parse.urlsplit(remote)
        remote_path = url.path.strip('/')
        object_store = get_object_store(remote)
        object_store.download_object(remote_path, local)
    elif remote.startswith('http://'):
        download_from_http(remote, local)
    else:
        download_from_local(remote, local)

    if compression_scheme is not None:
        tempfile = local_decompressed + '.tmp'
        if compression_scheme == 'gz':
            with gzip.open(local, 'rb') as gzipfile:
                with open(tempfile, 'xb') as dest_file:
                    shutil.copyfileobj(gzipfile, dest_file)
        else:
            raise NotImplementedError
        os.rename(tempfile, local_decompressed)
        os.remove(local)


def download_or_wait(remote: Optional[str],
                     local: str,
                     wait: bool = False,
                     max_retries: int = 2,
                     timeout: float = 60) -> None:
    """Downloads a file from remote to local, or waits for it to be downloaded.

    Does not do any thread safety checks, so we assume the calling function is using ``wait`` correctly.

    Args:
        remote (Optional[str]): Remote path (S3, SFTP, or local filesystem).
        local (str): Local path (local filesystem).
        wait (bool, default False): If ``true``, then do not actively download the file, but instead wait (up to
            ``timeout`` seconds) for the file to arrive.
        max_retries (int, default 2): Number of download re-attempts before giving up.
        timeout (float, default 60): How long to wait for file to download before raising an exception.
    """
    local_decompressed, _ = split_compression_suffix(local)
    last_error = None
    error_msgs = []
    for _ in range(1 + max_retries):
        try:
            if wait:
                start = time.time()
                while not os.path.exists(local_decompressed):
                    if time.time() - start > timeout:
                        raise TimeoutError(f'Waited longer than {timeout}s for other worker to download {local}.')
                    time.sleep(0.25)
            else:
                dispatch_download(remote, local)
            break
        except FileNotFoundError:
            raise  # bubble up file not found error
        except Exception as e:  # Retry for all causes of failure.
            error_msgs.append(e)
            last_error = e
            continue
    if len(error_msgs) > max_retries:
        raise RuntimeError(f'Failed to download {remote} -> {local}. Got errors:\n{error_msgs}') from last_error
