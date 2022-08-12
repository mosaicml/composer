# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Hyperparameters and registry for the :class:`.ObjectStore` implementations."""

import abc
import dataclasses
import os
from typing import Any, Dict, Optional, Type

import yahp as hp

from composer.utils.object_store.libcloud_object_store import LibcloudObjectStore
from composer.utils.object_store.object_store import ObjectStore
from composer.utils.object_store.s3_object_store import S3ObjectStore
from composer.utils.object_store.sftp_object_store import SFTPObjectStore

__all__ = [
    'ObjectStoreHparams', 'LibcloudObjectStoreHparams', 'S3ObjectStoreHparams', 'SFTPObjectStoreHparams',
    'object_store_registry'
]


@dataclasses.dataclass
class ObjectStoreHparams(hp.Hparams, abc.ABC):
    """Base class for :class:`.ObjectStore` hyperparameters."""

    @abc.abstractmethod
    def get_object_store_cls(self) -> Type[ObjectStore]:
        """Returns the type of :class:`.ObjectStore`."""
        raise NotImplementedError()

    @abc.abstractmethod
    def get_kwargs(self) -> Dict[str, Any]:
        """Returns the kwargs to construct the object store returned by :meth:`get_object_store_class`.

        Returns:
            Dict[str, Any]: The kwargs.
        """
        raise NotImplementedError()

    def initialize_object(self) -> ObjectStore:
        # error: Expected no arguments to "ObjectStore" constructor
        return self.get_object_store_cls()(**self.get_kwargs())  # type: ignore


@dataclasses.dataclass
class LibcloudObjectStoreHparams(ObjectStoreHparams):
    """:class:`~.LibcloudObjectStore` hyperparameters.

    .. rubric:: Example

    Here's an example on how to connect to an Amazon S3 bucket. This example assumes:

    * The container is named named ``MY_CONTAINER``.
    * The AWS Access Key ID is stored in an environment variable named ``AWS_ACCESS_KEY_ID``.
    * The Secret Access Key is in an environmental variable named ``AWS_SECRET_ACCESS_KEY``.

    .. testsetup:: composer.utils.object_store.object_store_hparams.__init__.s3

        import os

        os.environ["AWS_ACCESS_KEY_ID"] = "key"
        os.environ["AWS_SECRET_ACCESS_KEY"] = "secret"

    .. doctest:: composer.utils.object_store.object_store_hparams.__init__.s3

        >>> from composer.utils.object_store.object_store_hparams import LibcloudObjectStoreHparams
        >>> provider_hparams = LibcloudObjectStoreHparams(
        ...     provider="s3",
        ...     container="MY_CONTAINER",
        ...     key_environ="AWS_ACCESS_KEY_ID",
        ...     secret_environ="AWS_SECRET_ACCESS_KEY",
        ... )
        >>> provider = provider_hparams.initialize_object()
        >>> provider
        <composer.utils.object_store.libcloud_object_store.LibcloudObjectStore object at ...>

    Args:
        provider (str): Cloud provider to use.

            See :class:`LibcloudObjectStore` for documentation.
        container (str): The name of the container (i.e. bucket) to use.
        key_environ (str, optional): The name of an environment variable containing the API key or username
            to use to connect to the provider. If no key is required, then set this field to ``None``.
            (default: ``None``)

            For security reasons, composer requires that the key be specified via an environment variable.
            For example, if your key is an environment variable called ``OBJECT_STORE_KEY`` that is set to ``MY_KEY``,
            then you should set this parameter equal to ``OBJECT_STORE_KEY``. Composer will read the key like this:

            .. testsetup::  composer.utils.object_store.object_store_hparams.LibcloudObjectStoreHparams.__init__.key

                import os
                import functools
                from composer.utils.object_store.object_store_hparams import LibcloudObjectStoreHparams

                os.environ["OBJECT_STORE_KEY"] = "MY_KEY"
                LibcloudObjectStoreHparams = functools.partial(LibcloudObjectStoreHparams, provider="s3", container="container")

            .. doctest:: composer.utils.object_store.object_store_hparams.LibcloudObjectStoreHparams.__init__.key

                >>> import os
                >>> params = LibcloudObjectStoreHparams(key_environ="OBJECT_STORE_KEY")
                >>> key = os.environ[params.key_environ]
                >>> key
                'MY_KEY'

        secret_environ (str, optional): The name of an environment variable containing the API secret  or password
            to use for the provider. If no secret is required, then set this field to ``None``. (default: ``None``)

            For security reasons, composer requires that the secret be specified via an environment variable.
            For example, if your secret is an environment variable called ``OBJECT_STORE_SECRET`` that is set to ``MY_SECRET``,
            then you should set this parameter equal to ``OBJECT_STORE_SECRET``. Composer will read the secret like this:

            .. testsetup:: composer.utils.object_store.object_store_hparams.LibcloudObjectStoreHparams.__init__.secret

                import os
                import functools
                from composer.utils.object_store.object_store_hparams import LibcloudObjectStoreHparams

                original_secret = os.environ.get("OBJECT_STORE_SECRET")
                os.environ["OBJECT_STORE_SECRET"] = "MY_SECRET"
                LibcloudObjectStoreHparams = functools.partial(LibcloudObjectStoreHparams, provider="s3", container="container")


            .. doctest:: composer.utils.object_store.object_store_hparams.LibcloudObjectStoreHparams.__init__.secret

                >>> import os
                >>> params = LibcloudObjectStoreHparams(secret_environ="OBJECT_STORE_SECRET")
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

    provider: str = hp.auto(LibcloudObjectStore, 'provider')
    container: str = hp.auto(LibcloudObjectStore, 'container')
    key_environ: Optional[str] = hp.optional(('The name of an environment variable containing '
                                              'an API key or username to use to connect to the provider.'),
                                             default=None)
    secret_environ: Optional[str] = hp.optional(('The name of an environment variable containing '
                                                 'an API secret or password to use to connect to the provider.'),
                                                default=None)
    region: Optional[str] = hp.optional('Cloud region to use', default=None)
    host: Optional[str] = hp.optional('Override hostname for connections', default=None)
    port: Optional[int] = hp.optional('Override port for connections', default=None)
    extra_init_kwargs: Dict[str, Any] = hp.optional(
        'Extra keyword arguments to pass into the constructor for the specified provider.', default_factory=dict)

    def get_object_store_cls(self) -> Type[ObjectStore]:
        return LibcloudObjectStore

    def get_kwargs(self) -> Dict[str, Any]:
        init_kwargs = {
            'provider': self.provider,
            'container': self.container,
            'provider_kwargs': {},
        }
        for key in ('host', 'port', 'region'):
            kwarg = getattr(self, key)
            if getattr(self, key) is not None:
                init_kwargs['provider_kwargs'][key] = kwarg
        init_kwargs['provider_kwargs']['key'] = None if self.key_environ is None else os.environ[self.key_environ]
        init_kwargs['provider_kwargs']['secret'] = None if self.secret_environ is None else os.environ[
            self.secret_environ]
        init_kwargs.update(self.extra_init_kwargs)
        return init_kwargs


@dataclasses.dataclass
class S3ObjectStoreHparams(ObjectStoreHparams):
    """:class:`~.S3ObjectStore` hyperparameters.

    The :class:`.S3ObjectStore` uses :mod:`boto3` to handle uploading files to and downloading files from
    S3-Compatible object stores.

    .. note::

        To follow best security practices, credentials cannot be specified as part of the hyperparameters.
        Instead, please ensure that credentials are in the environment, which will be read automatically.

        See :ref:`guide to credentials <boto3:guide_credentials>` for more information.

    Args:
        bucket (str): See :class:`.S3ObjectStore`.
        prefix (str): See :class:`.S3ObjectStore`.
        region_name (str, optional): See :class:`.S3ObjectStore`.
        endpoint_url (str, optional): See :class:`.S3ObjectStore`.
        client_config (dict, optional): See :class:`.S3ObjectStore`.
        transfer_config (dict, optional): See :class:`.S3ObjectStore`.
    """

    bucket: str = hp.auto(S3ObjectStore, 'bucket')
    prefix: str = hp.auto(S3ObjectStore, 'prefix')
    region_name: Optional[str] = hp.auto(S3ObjectStore, 'region_name')
    endpoint_url: Optional[str] = hp.auto(S3ObjectStore, 'endpoint_url')
    # Not including the credentials as part of the hparams -- they should be specified through the default
    # environment variables
    client_config: Optional[Dict[Any, Any]] = hp.auto(S3ObjectStore, 'client_config')
    transfer_config: Optional[Dict[Any, Any]] = hp.auto(S3ObjectStore, 'transfer_config')

    def get_object_store_cls(self) -> Type[ObjectStore]:
        return S3ObjectStore

    def get_kwargs(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass
class SFTPObjectStoreHparams(ObjectStoreHparams):
    """:class:`~.SFTPObjectStore` hyperparameters.

    The :class:`.SFTPObjectStore` uses :mod:`paramiko` to upload and download files from SSH servers

    .. note::

        To follow best security practices, credentials shouldn't be specified as part of the hyperparameters.
        Instead, please ensure that credentials are in the environment, which will be read automatically.

    Args:
        host (str): See :class:`.SFTPObjectStore`.
        port (int, optional): See :class:`.SFTPObjectStore`.
        username (str, optional): See :class:`.SFTPObjectStore`.
        known_hosts_filename (str, optional): See :class:`.SFTPObjectStore`.
        key_filename (str, optional): See :class:`.SFTPObjectStore`.
        cwd (str, optional): See :class:`.SFTPObjectStore`.
        connect_kwargs (Dict[str, Any], optional): See :class:`.SFTPObjectStore`.
    """

    host: str = hp.auto(SFTPObjectStore, 'host')
    port: int = hp.auto(SFTPObjectStore, 'port')
    username: Optional[str] = hp.auto(SFTPObjectStore, 'username')
    known_hosts_filename: Optional[str] = hp.auto(SFTPObjectStore, 'known_hosts_filename')
    known_hosts_filename_environ: str = hp.optional(
        ('The name of an environment variable containing '
         'the path to a SSH known hosts file. Note that `known_hosts_filename` takes precedence over this variable.'),
        default='COMPOSER_SFTP_KNOWN_HOSTS_FILE',
    )
    key_filename: Optional[str] = hp.auto(SFTPObjectStore, 'key_filename')
    key_filename_environ: str = hp.optional(
        ('The name of an environment variable containing '
         'the path to a SSH keyfile. Note that `key_filename` takes precedence over this variable.'),
        default='COMPOSER_SFTP_KEY_FILE')
    missing_host_key_policy: str = hp.auto(SFTPObjectStore, 'missing_host_key_policy')
    cwd: str = hp.auto(SFTPObjectStore, 'cwd')
    connect_kwargs: Optional[Dict[str, Any]] = hp.auto(SFTPObjectStore, 'connect_kwargs')

    def get_object_store_cls(self) -> Type[ObjectStore]:
        return SFTPObjectStore

    def get_kwargs(self) -> Dict[str, Any]:
        kwargs = dataclasses.asdict(self)
        del kwargs['key_filename_environ']
        if self.key_filename_environ in os.environ and self.key_filename is None:
            kwargs['key_filename'] = os.environ[self.key_filename_environ]

        del kwargs['known_hosts_filename_environ']
        if self.known_hosts_filename_environ in os.environ and self.known_hosts_filename is None:
            kwargs['known_hosts_filename'] = os.environ[self.known_hosts_filename_environ]

        return kwargs


object_store_registry: Dict[str, Type[ObjectStoreHparams]] = {
    'libcloud': LibcloudObjectStoreHparams,
    's3': S3ObjectStoreHparams,
    'sftp': SFTPObjectStoreHparams,
}
