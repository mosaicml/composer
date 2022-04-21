import os
import shutil
from time import sleep, time
from typing import Optional
from urllib.parse import urlparse


def wait_for_download(local: str, timeout: Optional[float] = 10) -> None:
    """Block until another worker's shard download completes.

    Args:
        local (str): Path to file.
        timeout (Optional[float]): How long to wait before raising an exception. Default: 10 sec.
    """
    start_time = time()
    i = 0
    while True:
        if os.path.exists(local):
            return
        elapsed = time() - start_time
        if timeout is not None:
            assert elapsed < timeout, f'Waited too long (more than {timeout:.3f} sec) for download'
        sleep(0.1)
        i += 1


def download_from_s3(remote: str, local: str) -> None:
    """Download a file from remote to local.

    Args:
        remote (str): Remote path (S3).
        local (str): Local path (local filesystem).
    """
    import boto3
    obj = urlparse(remote)
    assert obj.scheme == 's3'
    s3 = boto3.client('s3')
    s3.download_file(obj.netloc, obj.path[1:], local)


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
    os.makedirs(local_dir, exist_ok=True)
    if remote.startswith('s3://'):
        download_from_s3(remote, local)
    else:
        download_from_local(remote, local)


def safe_download(remote: str, local: str, timeout: Optional[float] = 10) -> None:
    """Safely download a file from remote to local.

    Args:
        remote (str): Remote path (S3 or local filesystem).
        local (str): Local path (local filesystem).
        timeout (Optional[float]): How long to wait before raising an exception. Default: 10 sec.
    """
    # If we already have the file cached locally, we are done.
    if os.path.exists(local):
        return

    # No local file, so check to see if someone else is currently downloading
    # the shard. If they are, wait for that download to complete.
    local_tmp = local + '.tmp'
    if os.path.exists(local_tmp):
        wait_for_download(local, timeout)
        return

    # No temp download file when we checked, so attept to take it ourself. If
    # that fails, someone beat us to it.
    local_dir = os.path.dirname(local)
    os.makedirs(local_dir, exist_ok=True)
    try:
        with open(local_tmp, 'xb') as out:
            out.write(b'')
    except FileExistsError:
        wait_for_download(local, timeout)
        return

    # We took the temp download file. Perform the download, then rename.
    download(remote, local_tmp)
    os.rename(local_tmp, local)
