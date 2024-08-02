# Copyright 2024 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Useful functions for load checkpoints from remote object store or local disk."""

import logging
from typing import Optional

from composer.utils import (
    dist,
    extract_path_from_symlink,
    maybe_create_object_store_from_uri,
    parse_uri,
    retry,
)

log = logging.getLogger(__name__)


def download_file(
    source_uri: str,
    destination_path: str,
    node_ranks: Optional[list[int]] = None,
    num_attempts: int = 5,
):
    """Downloads a file (object) from the specified URI to the specified directory.

    Args:
        source_uri (str): The URI to download the file from or a symlink to the URI.
        destination_path (str): The directory to download the file to.
        node_ranks (list[int]): The ranks of the nodes that will download the file. If None, all nodes will download the file.
        num_attempts (int): Retry for object store downloads. Default to 5.
    """
    # Only local rank 0 downloads
    local_rank = dist.get_local_rank()
    if local_rank != 0:
        return

    node_rank = dist.get_node_rank()
    if node_ranks is not None and node_rank not in node_ranks:
        return

    if source_uri.endswith('.symlink'):
        source_path = download_and_extract_symlink(source_uri)

    object_store = maybe_create_object_store_from_uri(source_uri)
    _, _, source_path = parse_uri(source_uri)
    assert object_store is not None

    @retry(num_attempts=num_attempts)
    def _download():
        object_store.download_object(
            object_name=source_path,
            filename=destination_path,
        )

    log.debug(f'Downloading {source_path} to {destination_path}')
    _download()
    log.debug(f'Finished downloading {source_path} to {destination_path}')


def download_and_extract_symlink(source_uri) -> str:
    """Downloads a symlink file from the specified URI and returns the path it points to.

    Args:
        source_uri (str): The URI to download the symlink from.

    Returns:
        str: The path the symlink points to.
    """
    object_store = maybe_create_object_store_from_uri(source_uri)
    _, _, source_path = parse_uri(source_uri)
    source_path = extract_path_from_symlink(source_path, object_store)
    return source_path


def download_monolithic_checkpoint(
    source_uri: str,
    destination_path: str,
    global_rank_zero_only: bool = True,
):
    """Downloads a monolithic checkpoint from the specified URI to the specified directory.

    Args:
        source_uri (str): The URI to download the checkpoint from or symlink that points to the URI.
        destination_path (str): The directory to download the checkpoint to.
        global_rank_zero_only (bool): If True, only rank 0 will download the checkpoint.
        broadcast_files_to_other_nodes (bool): If True, the downloaded checkpoint will be broadcast to all other nodes.
            If torch syncs modules states this is unnecessary.
    """
    node_ranks = None
    if global_rank_zero_only:
        node_ranks = [0]
    download_file(
        source_uri=source_uri,
        destination_path=destination_path,
        node_ranks=node_ranks,
    )
    if not global_rank_zero_only or (global_rank_zero_only and dist.get_global_rank() == 0):
        return destination_path
    return None
