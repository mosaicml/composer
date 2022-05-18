# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Download handling for :class:`StreamingDataset`.
"""

import os
import shutil
import textwrap
from urllib.parse import urlparse

__all__ = ["safe_download"]


def download_from_s3(remote: str, local: str, timeout: float) -> None:
    """Download a file from remote to local.

    Args:
        remote (str): Remote path (S3).
        local (str): Local path (local filesystem).
        timeout (float): How long to wait for shard to download before raising an exception.
    """
    try:
        import boto3  # type: ignore (third-party)
        from botocore.config import Config  # type: ignore (third-party)
    except ImportError as e:
        raise ImportError(
            textwrap.dedent("""\
            Composer was installed without streaming support. To use streaming with Composer, run: `pip install mosaicml
            [streaming]` if using pip or `conda install -c conda-forge monai` if using Anaconda""")) from e

    obj = urlparse(remote)
    if obj.scheme != 's3':
        raise ValueError(f"Expected obj.scheme to be 's3', got {obj.scheme} for remote={remote}")

    config = Config(read_timeout=timeout)
    s3 = boto3.client('s3', config=config)
    local_tmp = local + '.tmp'
    try:
        s3.download_file(obj.netloc, obj.path[1:], local_tmp)
    except FileNotFoundError:
        if os.path.exists(local_tmp):
            os.remove(local_tmp)
        return
    if os.path.exists(local_tmp):
        if not os.path.exists(local):
            os.rename(local_tmp, local)
        else:
            os.remove(local_tmp)


def download_from_local(remote: str, local: str) -> None:
    """Download a file from remote to local.

    Args:
        remote (str): Remote path (local filesystem).
        local (str): Local path (local filesystem).
    """
    shutil.copy(remote, local)


def download(remote: str, local: str, timeout: float) -> None:
    """Download a file from remote to local.

    Args:
        remote (str): Remote path (S3 or local filesystem).
        local (str): Local path (local filesystem).
        timeout (float): How long to wait for shard to download before raising an exception.
    """
    local_dir = os.path.dirname(local)
    os.makedirs(local_dir, exist_ok=True)

    if remote.startswith('s3://'):
        download_from_s3(remote, local, timeout)
    else:
        download_from_local(remote, local)


def safe_download(remote: str, local: str, max_retries: int = 2, timeout: float = 60) -> None:
    """Safely downloads a file from remote to local.
       Handles multiple threads attempting to download the same shard.
       Gracefully deletes stale tmp files from crashed runs.

    Args:
        remote (str): Remote path (S3 or local filesystem).
        local (str): Local path (local filesystem).
        max_retries (int, default 2): Number of download attempts before giving up.
        timeout (float, default 60): How long to wait for shard to download before raising an exception.
    """
    if os.path.exists(local):
        return

    ok = False
    for _ in range(1 + max_retries):
        try:
            download(remote, local, timeout)
            ok = True
            break
        except:  # Retry for all causes of failure.
            pass
    assert ok
