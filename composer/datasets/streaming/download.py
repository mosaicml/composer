# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import os

__all__ = ["download"]


def _download_from_s3(remote: str, local: str, retry: int, timeout: float) -> None:
    """Download a file from S3.

    Args:
        remote (str): Remote file path.
        local (str): Local file path.
        retry (int): Number of retries.
        timeout (float): Connect and read timeout.
    """
    from urllib.parse import urlparse

    import boto3
    from botocore.config import Config
    obj = urlparse(remote)
    assert obj.scheme == 's3'
    config = Config(connect_timeout=timeout, read_timeout=timeout, retries={'total_max_attempts': retry + 1})
    s3 = boto3.client('s3', config=config)
    s3.download_file(obj.netloc, obj.path[1:], local)


def _download_from_local(remote: str, local: str) -> None:
    """Download a file from the local filesystem.

    Args:
        remote (str): Remote file path.
        local (str): Local file path.
    """
    import shutil
    shutil.copy(remote, local)


def download(remote: str, local: str, retry: int, timeout: float):
    """Download a file from a remote filesystem.

    Args:
        remote (str): Remote file path.
        local (str): Local file path.
        retry (int): Number of retries.
        timeout (float): Connect and read timeout.
    """
    if os.path.exists(local):
        return
    os.makedirs(os.path.dirname(local), exist_ok=True)
    if remote.startswith('s3://'):
        _download_from_s3(remote, local, retry, timeout)
    else:
        _download_from_local(remote, local)
