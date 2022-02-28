# Copyright 2021 MosaicML. All Rights Reserved.

"""Utility for uploading to and downloading from cloud object stores."""
import dataclasses
import os
import sys
import textwrap
from typing import Any, Dict, Iterator, Optional, Union

import yahp as hp
from libcloud.storage.providers import get_driver

__all__ = ["ObjectStoreProviderHparams", "ObjectStoreProvider"]


@dataclasses.dataclass
class ObjectStoreProviderHparams(hp.Hparams):
    """:class:`~composer.utils.object_store.ObjectStoreProvider` hyperparameters.

    .. rubric:: Example

    Here's an example on how to connect to an Amazon S3 bucket. This example assumes:

    * The container is named named ``MY_CONTAINER``.
    * The AWS Access Key ID is stored in an environment variable named ``AWS_ACCESS_KEY_ID``.
    * The Secret Access Key is in an environmental variable named ``AWS_SECRET_ACCESS_KEY``.

    .. testsetup:: composer.utils.object_store.ObjectStoreProviderHparams.__init__.s3

        import os

        os.environ["AWS_ACCESS_KEY_ID"] = "key"
        os.environ["AWS_SECRET_ACCESS_KEY"] = "secret"

    .. doctest:: composer.utils.object_store.ObjectStoreProviderHparams.__init__.s3

        >>> provider_hparams = ObjectStoreProviderHparams(
        ...     provider="s3",
        ...     container="MY_CONTAINER",
        ...     key_environ="AWS_ACCESS_KEY_ID",
        ...     secret_environ="AWS_SECRET_ACCESS_KEY",
        ... )
        >>> provider = provider_hparams.initialize_object()
        >>> provider
        <composer.utils.object_store.ObjectStoreProvider object at ...>

    Args:
        provider (str): Cloud provider to use.

            See :class:`ObjectStoreProvider` for documentation.
        container (str): The name of the container (i.e. bucket) to use.
        key_environ (str, optional): The name of an environment variable containing the API key or username
            to use to connect to the provider. If no key is required, then set this field to ``None``.
            (default: ``None``)
            
            For security reasons, composer requires that the key be specified via an environment variable.
            For example, if your key is an environment variable called ``OBJECT_STORE_KEY`` that is set to ``MY_KEY``,
            then you should set this parameter equal to ``OBJECT_STORE_KEY``. Composer will read the key like this:
            
            .. testsetup::  composer.utils.object_store.ObjectStoreProviderHparams.__init__.key

                import os
                import functools

                os.environ["OBJECT_STORE_KEY"] = "MY_KEY"
                ObjectStoreProviderHparams = functools.partial(ObjectStoreProviderHparams, provider="s3", container="container")
            
            .. doctest:: composer.utils.object_store.ObjectStoreProviderHparams.__init__.key

                >>> import os
                >>> params = ObjectStoreProviderHparams(key_environ="OBJECT_STORE_KEY")
                >>> key = os.environ[params.key_environ]
                >>> key
                'MY_KEY'

        secret_environ (str, optional): The name of an environment variable containing the API secret  or password
            to use for the provider. If no secret is required, then set this field to ``None``. (default: ``None``)

            For security reasons, composer requires that the secret be specified via an environment variable.
            For example, if your secret is an environment variable called ``OBJECT_STORE_SECRET`` that is set to ``MY_SECRET``,
            then you should set this parameter equal to ``OBJECT_STORE_SECRET``. Composer will read the secret like this:
                
            .. testsetup:: composer.utils.object_store.ObjectStoreProviderHparams.__init__.secret

                import os
                import functools
                original_secret = os.environ.get("OBJECT_STORE_SECRET")
                os.environ["OBJECT_STORE_SECRET"] = "MY_SECRET"
                ObjectStoreProviderHparams = functools.partial(ObjectStoreProviderHparams, provider="s3", container="container")


            .. doctest:: composer.utils.object_store.ObjectStoreProviderHparams.__init__.secret

                >>> import os
                >>> params = ObjectStoreProviderHparams(secret_environ="OBJECT_STORE_SECRET")
                >>> secret = os.environ[params.secret_environ]
                >>> secret
                'MY_SECRET'

        region (str, optional): Cloud region to use for the cloud provider.
            Most providers do not require the region to be specified. (default: ``None``)
        host (str, optional): Override the hostname for the cloud provider. (default: ``None``)
        port (int, optional): Override the port for the cloud provider. (default: ``None``)
        extra_init_kwargs (Dict[str, Any], optional): Extra keyword arguments to pass into the constructor
            for the specified provider. (default: ``None``, which is equivalent to an empty dictionary)

            .. seealso:: :class:`libcloud.storage.base.StorageDriver`

    """

    provider: str = hp.required("Cloud provider to use.")
    container: str = hp.required("The name of the container (i.e. bucket) to use.")
    key_environ: Optional[str] = hp.optional(textwrap.dedent("""\
        The name of an environment variable containing
        an API key or username to use to connect to the provider."""),
                                             default=None)
    secret_environ: Optional[str] = hp.optional(textwrap.dedent("""\
        The name of an environment variable containing
        an API secret or password to use to connect to the provider."""),
                                                default=None)
    region: Optional[str] = hp.optional("Cloud region to use", default=None)
    host: Optional[str] = hp.optional("Override hostname for connections", default=None)
    port: Optional[int] = hp.optional("Override port for connections", default=None)
    extra_init_kwargs: Dict[str, Any] = hp.optional(
        "Extra keyword arguments to pass into the constructor for the specified provider.", default_factory=dict)

    def initialize_object(self):
        """Returns an instance of :class:`ObjectStoreProvider`.

        Returns:
            ObjectStoreProvider: The provider.
        """
        init_kwargs = {}
        for key in ("host", "port", "region"):
            kwarg = getattr(self, key)
            if getattr(self, key) is not None:
                init_kwargs[key] = kwarg
        init_kwargs["key"] = None if self.key_environ is None else os.environ[self.key_environ]
        init_kwargs["secret"] = None if self.secret_environ is None else os.environ[self.secret_environ]
        init_kwargs.update(self.extra_init_kwargs)
        return ObjectStoreProvider(
            provider=self.provider,
            container=self.container,
            provider_init_kwargs=init_kwargs,
        )


class ObjectStoreProvider:
    """Utility for uploading to and downloading from object (blob) stores, such as Amazon S3.

    .. rubric:: Example

    Here's an example for an Amazon S3 bucket named ``MY_CONTAINER``:

    >>> provider = ObjectStoreProvider(
    ...     provider="s3",
    ...     container="MY_CONTAINER",
    ...     provider_init_kwargs={
    ...         "key": "AKIA...",
    ...         "secret": "*********",
    ...     }
    ... )
    >>> provider
    <composer.utils.object_store.ObjectStoreProvider object at ...>

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
        provider_init_kwargs (Dict[str, Any], optional):  Keyword arguments to pass into the constructor
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

    def __init__(self, provider: str, container: str, provider_init_kwargs: Optional[Dict[str, Any]] = None) -> None:
        provider_cls = get_driver(provider)
        if provider_init_kwargs is None:
            provider_init_kwargs = {}
        self._provider = provider_cls(**provider_init_kwargs)
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
        return self._provider.get_object(self._container.name, object_name)

    def get_object_size(self, object_name: str) -> int:
        """Get the size of an object, in bytes.

        Args:
            object_name (str): The name of the object.

        Returns:
            int: The object size, in bytes.
        """
        return self._get_object(object_name).size

    def download_object(self,
                        object_name: str,
                        destination_path: str,
                        overwrite_existing: bool = False,
                        delete_on_failure: bool = True):
        """Download an object to the specified destination path.

        .. seealso:: :meth:`libcloud.storage.base.StorageDriver.download_object`.

        Args:
            object_name (str): The name of the object to download.

            destination_path (str): Full path to a file or a directory where the incoming file will be saved.

            overwrite_existing (bool, optional): Set to ``True`` to overwrite an existing file. (default: ``False``)
            delete_on_failure (bool, optional): Set to ``True`` to delete a partially downloaded file if
                the download was not successful (hash mismatch / file size). (default: ``True``)
        """
        obj = self._get_object(object_name)
        self._provider.download_object(obj=obj,
                                       destination_path=destination_path,
                                       overwrite_existing=overwrite_existing,
                                       delete_on_failure=delete_on_failure)

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
