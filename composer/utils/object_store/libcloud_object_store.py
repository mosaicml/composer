# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Utility for uploading to and downloading from cloud object stores."""
import io
import os
import pathlib
import uuid
from typing import Any, Callable, Optional, Union

from requests.exceptions import ConnectionError
from urllib3.exceptions import ProtocolError

from composer.utils.import_helpers import MissingConditionalImportError
from composer.utils.iter_helpers import iterate_with_callback
from composer.utils.object_store.object_store import ObjectStore, ObjectStoreTransientError

__all__ = ['LibcloudObjectStore']


class LibcloudObjectStore(ObjectStore):
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
    <composer.utils.object_store.libcloud_object_store.LibcloudObjectStore object at ...>

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
        provider_kwargs (dict[str, Any], optional):  Keyword arguments to pass into the constructor
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

        key_environ (str, optional): Environment variable name for the API Key. Only used
            if 'key' is not in ``provider_kwargs``. Default: None.
        secret_environ (str, optional): Envrionment varaible for the Secret password. Only
            used if 'secret' is not in ``provider_kwargs``. Default: None.
    """

    def __init__(
        self,
        provider: str,
        container: str,
        chunk_size: int = 1_024 * 1_024,
        key_environ: Optional[str] = None,
        secret_environ: Optional[str] = None,
        provider_kwargs: Optional[dict[str, Any]] = None,
    ) -> None:
        try:
            from libcloud.storage.providers import get_driver
        except ImportError as e:
            raise MissingConditionalImportError('libcloud', 'apache-libcloud') from e
        provider_cls = get_driver(provider)
        if provider_kwargs is None:
            provider_kwargs = {}

        if 'key' not in provider_kwargs and \
           key_environ and key_environ in os.environ:
            provider_kwargs['key'] = os.environ[key_environ]

        if 'secret' not in provider_kwargs and \
           secret_environ and secret_environ in os.environ:
            provider_kwargs['secret'] = os.environ[secret_environ]

        self.chunk_size = chunk_size
        self._provider_name = provider
        self._provider = provider_cls(**provider_kwargs)
        self._container = self._provider.get_container(container)

    def get_uri(self, object_name: str):
        return f'{self._provider_name}://{self._container.name}/{object_name}'

    def upload_object(
        self,
        object_name: str,
        filename: Union[str, pathlib.Path],
        callback: Optional[Callable[[int, int], None]] = None,
    ):
        with open(filename, 'rb') as f:
            stream = iterate_with_callback(
                _file_to_iterator(f, self.chunk_size),
                os.fstat(f.fileno()).st_size,
                callback,
            )
            try:
                self._provider.upload_object_via_stream(
                    stream,
                    container=self._container,
                    object_name=object_name,
                )
            except Exception as e:
                self._ensure_transient_errors_are_wrapped(e)

    def _ensure_transient_errors_are_wrapped(self, exc: Exception):
        from libcloud.common.types import LibcloudError
        if isinstance(exc, (LibcloudError, ProtocolError, TimeoutError, ConnectionError)):
            if isinstance(exc, LibcloudError):
                # The S3 driver does not encode the error code in an easy-to-parse manner
                # So first checking if the error code is non-transient
                is_transient_error = any(x in str(exc) for x in ('408', '409', '425', '429', '500', '503', '504'))
                if not is_transient_error:
                    raise exc
            raise ObjectStoreTransientError() from exc
        raise exc

    def _get_object(self, object_name: str):
        """Get object from object store.

        Args:
            object_name (str): The name of the object.
        """
        from libcloud.storage.types import ObjectDoesNotExistError
        try:
            return self._provider.get_object(self._container.name, object_name)
        except ObjectDoesNotExistError as e:
            raise FileNotFoundError(f'Object not found: {self.get_uri(object_name)}') from e
        except Exception as e:
            self._ensure_transient_errors_are_wrapped(e)

    def get_object_size(self, object_name: str) -> int:
        obj = self._get_object(object_name)
        assert obj is not None
        return obj.size

    def download_object(
        self,
        object_name: str,
        filename: Union[str, pathlib.Path],
        overwrite: bool = False,
        callback: Optional[Callable[[int, int], None]] = None,
    ):
        if os.path.exists(filename) and not overwrite:
            # If the file already exits, short-circuit and skip the download
            raise FileExistsError(f'filename {filename} exists and overwrite was set to False.')

        dirname = os.path.dirname(filename)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        obj = self._get_object(object_name)
        # Download first to a tempfile, and then rename, in case if the file gets corrupted in transit
        tmp_filepath = str(filename) + f'.{uuid.uuid4()}.tmp'
        try:
            with open(tmp_filepath, 'wb+') as f:
                assert obj is not None
                stream = self._provider.download_object_as_stream(obj, chunk_size=self.chunk_size)
                for chunk in iterate_with_callback(stream, obj.size, callback):
                    f.write(chunk)
        except Exception as e:
            # The download failed for some reason. Make a best-effort attempt to remove the temporary file.
            try:
                os.remove(tmp_filepath)
            except OSError:
                pass
            self._ensure_transient_errors_are_wrapped(e)

        # The download was successful.
        if overwrite:
            os.replace(tmp_filepath, filename)
        else:
            os.rename(tmp_filepath, filename)

    def list_objects(self, prefix: Optional[str] = None) -> list[str]:
        if prefix is None:
            prefix = ''

        return [obj.name for obj in self._provider.list_container_objects(self._container, prefix=prefix)]


def _file_to_iterator(f: io.IOBase, chunk_size: int):
    while True:
        byte = f.read(chunk_size)
        if byte == b'':
            break
        yield byte
