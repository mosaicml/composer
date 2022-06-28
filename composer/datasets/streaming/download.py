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
from composer.utils import MissingConditionalImportError
# TODO: refactor to use object store for download, until then, use this private method.
from composer.utils.object_store.s3_object_store import _ensure_not_found_errors_are_wrapped

__all__ = ['download_or_wait']


def download_from_s3(remote: str, local: str, timeout: float) -> None:
    """Download a file from remote to local.

    Args:
        remote (str): Remote path (S3).
        local (str): Local path (local filesystem).
        timeout (float): How long to wait for shard to download before raising an exception.
    """
    try:
        import boto3
        from botocore.config import Config
        from botocore.exceptions import ClientError
    except ImportError as e:
        raise MissingConditionalImportError(extra_deps_group='streaming', conda_package='boto3') from e

    obj = urllib.parse.urlparse(remote)
    if obj.scheme != 's3':
        raise ValueError(f"Expected obj.scheme to be 's3', got {obj.scheme} for remote={remote}")

    config = Config(read_timeout=timeout)
    s3 = boto3.client('s3', config=config)
    try:
        s3.download_file(obj.netloc, obj.path.lstrip('/'), local)
    except ClientError as e:
        _ensure_not_found_errors_are_wrapped(remote, e)


def download_from_sftp(remote: str, local: str) -> None:
    """Download a file from remote SFTP server to local filepath.

    Authentication must be provided via username/password in the ``remote`` URI, or a valid SSH config, or a default key
    discoverable in ``~/.ssh/``.

    Args:
        remote (str): Remote path (SFTP).
        local (str): Local path (local filesystem).
    """
    try:
        from paramiko import SSHClient
    except ImportError as e:
        raise MissingConditionalImportError(extra_deps_group='streaming', conda_package='paramiko') from e

    # Parse URL
    url = urllib.parse.urlsplit(remote)
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
    remote_path = url.path

    # Get SSH key file if specified
    key_filename = os.environ.get('COMPOSER_SFTP_KEY_FILE', None)
    known_hosts_filename = os.environ.get('COMPOSER_SFTP_KNOWN_HOSTS_FILE', None)

    # Default port
    port = port if port else 22

    # Local tmp
    local_tmp = local + '.tmp'
    if os.path.exists(local_tmp):
        os.remove(local_tmp)

    with SSHClient() as ssh_client:
        # Connect SSH Client
        ssh_client.load_system_host_keys(known_hosts_filename)
        ssh_client.connect(
            hostname=hostname,
            port=port,
            username=username,
            password=password,
            key_filename=key_filename,
        )

        # SFTP Client
        sftp_client = ssh_client.open_sftp()
        sftp_client.get(remotepath=remote_path, localpath=local_tmp)
    os.rename(local_tmp, local)


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


def dispatch_download(remote: Optional[str], local: str, timeout: float):
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
    elif remote.startswith('s3://'):
        download_from_s3(remote, local, timeout)
    elif remote.startswith('sftp://'):
        download_from_sftp(remote, local)
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
                dispatch_download(remote, local, timeout=timeout)
            break
        except FileNotFoundError:
            raise  # bubble up file not found error
        except Exception as e:  # Retry for all causes of failure.
            error_msgs.append(e)
            last_error = e
            continue
    if len(error_msgs) > max_retries:
        raise RuntimeError(f'Failed to download {remote} -> {local}. Got errors:\n{error_msgs}') from last_error
