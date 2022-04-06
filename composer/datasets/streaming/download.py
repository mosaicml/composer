import os
import shutil

import boto3


def download(remote: str, local: str) -> None:
    """Universal downloader.

    Args:
        remote (str): Remote path (S3 or local filesystem).
        local (str): Local path (local filesystem).
    """
    dirname = os.path.dirname(local)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    if remote.startswith('s3://'):
        remote = remote[5:]
        idx = remote.index('/')
        bucket = remote[:idx]
        path = remote[idx + 1:]
        s3 = boto3.client('s3')
        s3.download_file(bucket, path, local)
    else:
        shutil.copy(remote, local)
