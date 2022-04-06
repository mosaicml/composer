import os
import shutil
from time import sleep

import boto3


def wait_for_download(local: str) -> None:
    """Block until another worker's shard download completes.

    Args:
        local (str): Path to file.
    """
    i = 0
    while True:
        if os.path.exists(local):
            return
        if 4 <= i and not i % 4:
            print('Waiting for download:', local)
        sleep(0.25)
        i += 1


def download_from_s3(remote: str, local: str) -> None:
    """Download a file from remote to local.

    Args:
        remote (str): Remote path (S3).
        local (str): Local path (local filesystem).
    """
    assert remote.startswith('s3://')
    remote = remote[5:]
    idx = remote.index('/')
    bucket = remote[:idx]
    path = remote[idx + 1:]
    s3 = boto3.client('s3')
    s3.download_file(bucket, path, local)


def download_from_local(remote: str, local: str) -> None:
    """Download a file from remote to local.

    Args:
        remote (str): Remote path (local filesystem).
        local (str): Local path (local filesystem).
    """
    shutil.copy(remote, local)


def download(remote: str, local: str) -> None:
    """Download a file from remote to local.

    Args:
        remote (str): Remote path (S3 or local filesystem).
        local (str): Local path (local filesystem).
    """
    local_dir = os.path.dirname(local)
    if not os.path.exists(local_dir):
        os.makedirs(local_dir)
    if remote.startswith('s3://'):
        download_from_s3(remote, local)
    else:
        download_from_local(remote, local)


def safe_download(remote: str, local: str) -> None:
    """Safely download a file from remote to local.

    Args:
        remote (str): Remote path (S3 or local filesystem).
        local (str): Local path (local filesystem).
    """
    # If we already have the file cached locally, we're done.
    if os.path.exists(local):
        return

    # Else if someone else is currently downloading the shard, wait for that download to complete.
    local_tmp = local + '.tmp'
    if os.path.exists(local_tmp):
        wait_for_download(local)
        return

    # Else if no one is downloading it, mark as in progress, then do the download ourself.
    local_dir = os.path.dirname(local)
    if not os.path.exists(local_dir):
        os.makedirs(local_dir)
    with open(local_tmp, 'w') as out:
        out.write('')
    download(remote, local_tmp)
    os.rename(local_tmp, local)
