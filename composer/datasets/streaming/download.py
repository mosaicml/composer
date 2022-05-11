# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Download handling for :class:`StreamingDataset`.
"""

import os
import shutil
import tempfile
import textwrap
from time import sleep, time
from urllib.parse import urlparse

__all__ = ["safe_download"]


def wait_for_download(local: str, timeout: float = 20) -> None:
    """Block until another worker's shard download completes.

    Args:
        local (str): Path to file.
        timeout (float): How long to wait before raising an exception. Default: 20 sec.
    """
    start_time = time()
    while True:
        if os.path.exists(local):
            return
        elapsed = time() - start_time
        if elapsed > timeout:
            raise TimeoutError(f'Waited too long (more than {timeout:.3f} sec) for download')
        sleep(0.1)


def download_from_s3(remote: str, local: str, timeout: float) -> None:
    """Download a file from remote to local.

    Args:
        remote (str): Remote path (S3).
        local (str): Local path (local filesystem).
        timeout (float): How long to wait for shard to download before raising an exception.
    """
    try:
        import boto3  # type: ignore (third-party)
        from botocore import Config  # type: ignore (third-party)
    except ImportError as e:
        raise ImportError(
            textwrap.dedent("""\
            Composer was installed without streaming support. To use streaming with Composer, run: `pip install mosaicml
            [streaming]` if using pip or `conda install -c conda-forge monai` if using Anaconda""")) from e

    obj = urlparse(remote)
    if obj.scheme != 's3':
        raise ValueError(f"Expected obj.scheme to be 's3', got {obj.scheme} for remote={remote}")

    # We don't know how much of total 'timeout' to assign to connect vs. read
    # So we allow both connect and read to take up to 'timeout' seconds
    # And if the overall time is greater than 'timeout', our parent `download` function will catch it.
    config = Config(connect_timeout=timeout, read_timeout=timeout, retries={'total_max_attempts': 5})
    s3 = boto3.client('s3', config=config)
    s3.download_file(obj.netloc, obj.path[1:], local)


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
    start_time = time()

    if remote.startswith('s3://'):
        download_from_s3(remote, local, timeout=timeout)
    else:
        download_from_local(remote, local)

    elapsed = time() - start_time
    if elapsed > timeout:
        raise TimeoutError(f'Waited too long (more than {timeout:.3f} sec) for download')


def safe_download(remote: str, local: str, timeout: float = 20) -> None:
    """Safely downloads a file from remote to local.
       Handles multiple threads attempting to download the same shard.
       Gracefully deletes stale tmp files from crashed runs.


    Args:
        remote (str): Remote path (S3 or local filesystem).
        local (str): Local path (local filesystem).
        timeout (float): How long to wait for shard to download before raising an exception. Default: 20 sec.
    """
    # If we already have the file cached locally, we are done.
    if os.path.exists(local):
        return

    # Check if there is a tmp file.
    local_tmp = local + '.tmp'
    if os.path.exists(local_tmp):
        # Get tmp file created time
        local_tmp_create_time = os.path.getctime(local_tmp)

        # Get current disk time, more consistent than system time
        with tempfile.NamedTemporaryFile() as f:
            current_disk_time = os.path.getctime(f.name)

        if current_disk_time - local_tmp_create_time < timeout + 1:  # 1s buffer to avoid race condition
            # If the tmp file is recent, it is either (1) from a very recent crashed run, or (2) another thread is actively downloading it.
            # So we wait but don't error out, in case we are in situation (1)
            try:
                wait_for_download(local, timeout)
                return
            except TimeoutError:
                pass

        # The tmp file is old, it is either (1) from a crashed run or (2) another thread is downloading it but is taking too long, and will timeout.
        # Let's delete the tmp file. If situation (1), this is safe. If situation (2), the other thread is expected to crash with a TimeoutError anyways, so this is fine.
        try:
            os.remove(local_tmp)
        except OSError:
            # This occurs if another download thread got to the delete first.
            pass

    # There is no tmp file, so attempt to make it.
    # If this fails, another download thread beat us to it, so wait.
    local_dir = os.path.dirname(local)
    os.makedirs(local_dir, exist_ok=True)
    try:
        with open(local_tmp, 'xb') as out:
            out.write(b'')
    except FileExistsError:
        # If we run out of time here, we know a download thread was active and exceeded timeout, so we should error out.
        wait_for_download(local, timeout)
        return

    # We succesfully created the tmp file. Perform the download and rename.
    download(remote, local_tmp, timeout)
    os.rename(local_tmp, local)
