# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Utility for uploading to and downloading from cloud object stores."""
import os
import sys
import tempfile
import uuid
from typing import Any, Dict, Iterator, Optional, Union

from libcloud.storage.providers import get_driver
from libcloud.storage.types import ObjectDoesNotExistError

__all__ = ["LibcloudObjectStore"]


class LibcloudObjectStore:
    """Utility for uploading to and downloading from object (blob) stores, such as Amazon S3.

    .. rubric:: Example

    Here's an example for an Amazon S3 bucket named ``MY_CONTAINER``:

    >>> from composer.utils import LibcloudObjectStore
    >>> object_store = LibcloudObjectStore(
    ...     provider="s3",
    ...     container="MY_CONTAINER",
    ...     provider_kwargs={
    ...         "key": "AKIA...",
    ...         "secret": "*********",
    ...     }
    ... )
    >>> object_store
    <composer.utils.libcloud_object_store.LibcloudObjectStore object at ...>

    Args:
        provider (str): Cloud provider to use. Valid options are:

            * :mod:`~libcloud.storage.drivers.atmos`
            * :mod:`~libcloud.storage.drivers.auroraobjects`
            * :mod:`~libcloud.storage.drivers.azure_blobs`
            * :mod:`~libcloud.storage.drivers.backblaze_b2`
            * :mod:`~libcloud.storage.drivers.cloudfiles`
            * :mod:`~libcloud.storage.drivers.digitalocean_spaces`
            * :mod:`~libcloud.storage.drivers.google_storage`
            * :mod:`~libcloud.storage.drivers.ktucloud`
            * :mod:`~libcloud.storage.drivers.local`
            * :mod:`~libcloud.storage.drivers.minio`
            * :mod:`~libcloud.storage.drivers.nimbus`
            * :mod:`~libcloud.storage.drivers.ninefold`
            * :mod:`~libcloud.storage.drivers.oss`
            * :mod:`~libcloud.storage.drivers.rgw`
            * :mod:`~libcloud.storage.drivers.s3`

            .. seealso:: :doc:`Full list of libcloud providers <libcloud:storage/supported_providers>`

        container (str): The name of the container (i.e. bucket) to use.
        provider_kwargs (Dict[str, Any], optional):  Keyword arguments to pass into the constructor
            for the specified provider. These arguments would usually include the cloud region
            and credentials.

            Common keys are:

            * ``key`` (str): API key or username to be used (required).
            * ``secret`` (str): Secret password to be used (required).
            * ``secure`` (bool): Whether to use HTTPS or HTTP. Note: Some providers only support HTTPS, and it is on by default.
            * ``host`` (str): Override hostname used for connections.
            * ``port`` (int): Override port used for connections.
            * ``api_version`` (str): Optional API version. Only used by drivers which support multiple API versions.
            * ``region`` (str): Optional driver region. Only used by drivers which support multiple regions.

            .. seealso:: :class:`libcloud.storage.base.StorageDriver`
    """

    def __init__(self, provider: str, container: str, provider_kwargs: Optional[Dict[str, Any]] = None) -> None:
        provider_cls = get_driver(provider)
        if provider_kwargs is None:
            provider_kwargs = {}
        self._provider = provider_cls(**provider_kwargs)
        self._container = self._provider.get_container(container)

    @property
    def provider_name(self):
        """The name of the cloud provider."""
        return self._provider.name

    @property
    def container_name(self):
        """The name of the object storage container."""
        return self._container.name

    def upload_object(self,
                      file_path: str,
                      object_name: str,
                      verify_hash: bool = True,
                      extra: Optional[Dict] = None,
                      headers: Optional[Dict[str, str]] = None):
        """Upload an object currently located on a disk.

        .. seealso:: :meth:`libcloud.storage.base.StorageDriver.upload_object`.

        Args:
            file_path (str): Path to the object on disk.
            object_name (str): Object name (i.e. where the object will be stored in the container.)
            verify_hash (bool, optional): Whether to verify hashes (default: ``True``)
            extra (Optional[Dict], optional): Extra attributes to pass to the underlying provider driver.
                (default: ``None``, which is equivalent to an empty dictionary)
            headers (Optional[Dict[str, str]], optional): Additional request headers, such as CORS headers.
                (defaults: ``None``, which is equivalent to an empty dictionary)
        """
        self._provider.upload_object(file_path=file_path,
                                     container=self._container,
                                     object_name=object_name,
                                     extra=extra,
                                     verify_hash=verify_hash,
                                     headers=headers)

    def upload_object_via_stream(self,
                                 obj: Union[bytes, Iterator[bytes]],
                                 object_name: str,
                                 extra: Optional[Dict] = None,
                                 headers: Optional[Dict[str, str]] = None):
        """Upload an object.

        .. seealso:: :meth:`libcloud.storage.base.StorageDriver.upload_object_via_stream`.

        Args:
            obj (bytes | Iterator[bytes]): The object.
            object_name (str): Object name (i.e. where the object will be stored in the container.)
            verify_hash (bool, optional): Whether to verify hashes (default: ``True``)
            extra (Optional[Dict], optional): Extra attributes to pass to the underlying provider driver.
                (default: ``None``)
            headers (Optional[Dict[str, str]], optional): Additional request headers, such as CORS headers.
                (defaults: ``None``)
        """
        if isinstance(obj, bytes):
            obj = iter(i.to_bytes(1, sys.byteorder) for i in obj)
        self._provider.upload_object_via_stream(iterator=obj,
                                                container=self._container,
                                                object_name=object_name,
                                                extra=extra,
                                                headers=headers)

    def _get_object(self, object_name: str):
        """Get object from object store.

        Recursively follow any symlinks. If an object does not exist, automatically
        checks if it is a symlink by appending ``.symlink``.

        Args:
            object_name (str): The name of the object.
        """
        obj = None
        try:
            obj = self._provider.get_object(self._container.name, object_name)
        except ObjectDoesNotExistError:
            # Object not found, check for potential symlink
            object_name += ".symlink"
            obj = self._provider.get_object(self._container.name, object_name)
        # Recursively trace any symlinks
        if obj.name.endswith(".symlink"):
            # Download symlink object to temporary folder
            with tempfile.TemporaryDirectory() as tmpdir:
                tmppath = os.path.join(tmpdir, str(uuid.uuid4()))
                self._provider.download_object(obj=obj,
                                               destination_path=tmppath,
                                               overwrite_existing=True,
                                               delete_on_failure=True)
                # Read object name in symlink and recurse
                with open(tmppath) as f:
                    symlinked_object_name = f.read()
                    return self._get_object(symlinked_object_name)
        return obj

    def get_object_size(self, object_name: str) -> int:
        """Get the size of an object, in bytes.

        Args:
            object_name (str): The name of the object.

        Returns:
            int: The object size, in bytes.
        """
        return self._get_object(object_name).size

    def download_object(
        self,
        object_name: str,
        destination_path: str,
        overwrite_existing: bool = False,
    ):
        """Download an object to the specified destination path.

        .. seealso:: :meth:`libcloud.storage.base.StorageDriver.download_object`.

        Args:
            object_name (str): The name of the object to download.
            destination_path (str): Full path to a file or a directory where the incoming file will be saved.
            overwrite_existing (bool, optional): Set to ``True`` to overwrite an existing file. (default: ``False``)
        """
        if os.path.exists(destination_path) and not overwrite_existing:
            # If the file already exits, short-circuit and skip the download
            raise FileExistsError(
                f"destination_path {destination_path} exists and overwrite_existing was set to False.")

        obj = self._get_object(object_name)
        # Download first to a tempfile, and then rename, in case if the file gets corrupted in transit
        tmp_filepath = destination_path + f".{uuid.uuid4()}.tmp"
        try:
            self._provider.download_object(
                obj=obj,
                destination_path=tmp_filepath,
            )
        except:
            # The download failed for some reason. Make a best-effort attempt to remove the temporary file.
            try:
                os.remove(tmp_filepath)
            except OSError:
                pass
            raise

        # The download was successful.
        if overwrite_existing:
            os.replace(tmp_filepath, destination_path)
        else:
            os.rename(tmp_filepath, destination_path)

    def download_object_as_stream(self, object_name: str, chunk_size: Optional[int] = None):
        """Return a iterator which yields object data.

        .. seealso:: :meth:`libcloud.storage.base.StorageDriver.download_object_as_stream`.

        Args:
            object_name (str): Object name.
            chunk_size (Optional[int], optional): Optional chunk size (in bytes).

        Returns:
            Iterator[bytes]: The object, as a byte stream.
        """
        obj = self._get_object(object_name)
        return self._provider.download_object_as_stream(obj, chunk_size=chunk_size)
