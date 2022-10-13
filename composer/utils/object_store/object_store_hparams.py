# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Hyperparameters and registry for the :class:`.RemoteFilesystem` implementations."""

import abc
import dataclasses
from typing import Any, Dict, Optional, Type

import yahp as hp

from composer.utils.object_store.libcloud_object_store import LibcloudRemoteFilesystem
from composer.utils.object_store.object_store import RemoteFilesystem
from composer.utils.object_store.s3_object_store import S3RemoteFilesystem
from composer.utils.object_store.sftp_object_store import SFTPRemoteFilesystem

__all__ = [
    'RemoteFilesystemHparams', 'LibcloudRemoteFilesystemHparams', 'S3RemoteFilesystemHparams',
    'SFTPRemoteFilesystemHparams', 'object_store_registry'
]


@dataclasses.dataclass
class RemoteFilesystemHparams(hp.Hparams, abc.ABC):
    """Base class for :class:`.RemoteFilesystem` hyperparameters."""

    @abc.abstractmethod
    def get_object_store_cls(self) -> Type[RemoteFilesystem]:
        """Returns the type of :class:`.RemoteFilesystem`."""
        raise NotImplementedError()

    @abc.abstractmethod
    def get_kwargs(self) -> Dict[str, Any]:
        """Returns the kwargs to construct the object store returned by :meth:`get_object_store_class`.

        Returns:
            Dict[str, Any]: The kwargs.
        """
        raise NotImplementedError()

    def initialize_object(self) -> RemoteFilesystem:
        # error: Expected no arguments to "RemoteFilesystem" constructor
        return self.get_object_store_cls()(**self.get_kwargs())  # type: ignore


@dataclasses.dataclass
class LibcloudRemoteFilesystemHparams(RemoteFilesystemHparams):
    """:class:`~.LibcloudRemoteFilesystem` hyperparameters.

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

        >>> from composer.utils.object_store.object_store_hparams import LibcloudRemoteFilesystemHparams
        >>> provider_hparams = LibcloudRemoteFilesystemHparams(
        ...     provider="s3",
        ...     container="MY_CONTAINER",
        ...     key_environ="AWS_ACCESS_KEY_ID",
        ...     secret_environ="AWS_SECRET_ACCESS_KEY",
        ... )
        >>> provider = provider_hparams.initialize_object()
        >>> provider
        <composer.utils.object_store.libcloud_object_store.LibcloudRemoteFilesystem object at ...>

    Args:
        provider (str): Cloud provider to use.

            See :class:`LibcloudRemoteFilesystem` for documentation.
        container (str): The name of the container (i.e. bucket) to use.
        key_environ (str, optional): The name of an environment variable containing the API key or username
            to use to connect to the provider. If no key is required, then set this field to ``None``.
            (default: ``None``)

            For security reasons, composer requires that the key be specified via an environment variable.
            For example, if your key is an environment variable called ``OBJECT_STORE_KEY`` that is set to ``MY_KEY``,
            then you should set this parameter equal to ``OBJECT_STORE_KEY``. Composer will read the key like this:

            .. testsetup::  composer.utils.object_store.object_store_hparams.LibcloudRemoteFilesystemHparams.__init__.key

                import os
                import functools
                from composer.utils.object_store.object_store_hparams import LibcloudRemoteFilesystemHparams

                os.environ["OBJECT_STORE_KEY"] = "MY_KEY"
                LibcloudRemoteFilesystemHparams = functools.partial(LibcloudRemoteFilesystemHparams, provider="s3", container="container")

            .. doctest:: composer.utils.object_store.object_store_hparams.LibcloudRemoteFilesystemHparams.__init__.key

                >>> import os
                >>> params = LibcloudRemoteFilesystemHparams(key_environ="OBJECT_STORE_KEY")
                >>> key = os.environ[params.key_environ]
                >>> key
                'MY_KEY'

        secret_environ (str, optional): The name of an environment variable containing the API secret  or password
            to use for the provider. If no secret is required, then set this field to ``None``. (default: ``None``)

            For security reasons, composer requires that the secret be specified via an environment variable.
            For example, if your secret is an environment variable called ``OBJECT_STORE_SECRET`` that is set to ``MY_SECRET``,
            then you should set this parameter equal to ``OBJECT_STORE_SECRET``. Composer will read the secret like this:

            .. testsetup:: composer.utils.object_store.object_store_hparams.LibcloudRemoteFilesystemHparams.__init__.secret

                import os
                import functools
                from composer.utils.object_store.object_store_hparams import LibcloudRemoteFilesystemHparams

                original_secret = os.environ.get("OBJECT_STORE_SECRET")
                os.environ["OBJECT_STORE_SECRET"] = "MY_SECRET"
                LibcloudRemoteFilesystemHparams = functools.partial(LibcloudRemoteFilesystemHparams, provider="s3", container="container")


            .. doctest:: composer.utils.object_store.object_store_hparams.LibcloudRemoteFilesystemHparams.__init__.secret

                >>> import os
                >>> params = LibcloudRemoteFilesystemHparams(secret_environ="OBJECT_STORE_SECRET")
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

    provider: str = hp.auto(LibcloudRemoteFilesystem, 'provider')
    container: str = hp.auto(LibcloudRemoteFilesystem, 'container')
    chunk_size: int = hp.optional('Chunk size of download/updates', default=1024 * 1024)
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

    def get_object_store_cls(self) -> Type[RemoteFilesystem]:
        return LibcloudRemoteFilesystem

    def get_kwargs(self) -> Dict[str, Any]:

        provider_kwargs = {}
        if self.region:
            provider_kwargs['region'] = self.region
        if self.host:
            provider_kwargs['host'] = self.host
        if self.port:
            provider_kwargs['port'] = self.port
        provider_kwargs.update(self.extra_init_kwargs)

        return {
            'provider': self.provider,
            'container': self.container,
            'chunk_size': self.chunk_size,
            'key_environ': self.key_environ,
            'secret_environ': self.secret_environ,
            'provider_kwargs': provider_kwargs,
        }


@dataclasses.dataclass
class S3RemoteFilesystemHparams(RemoteFilesystemHparams):
    """:class:`~.S3RemoteFilesystem` hyperparameters.

    The :class:`.S3RemoteFilesystem` uses :mod:`boto3` to handle uploading files to and downloading files from
    S3-Compatible object stores.

    .. note::

        To follow best security practices, credentials cannot be specified as part of the hyperparameters.
        Instead, please ensure that credentials are in the environment, which will be read automatically.

        See :ref:`guide to credentials <boto3:guide_credentials>` for more information.

    Args:
        bucket (str): See :class:`.S3RemoteFilesystem`.
        prefix (str): See :class:`.S3RemoteFilesystem`.
        region_name (str, optional): See :class:`.S3RemoteFilesystem`.
        endpoint_url (str, optional): See :class:`.S3RemoteFilesystem`.
        client_config (dict, optional): See :class:`.S3RemoteFilesystem`.
        transfer_config (dict, optional): See :class:`.S3RemoteFilesystem`.
    """

    bucket: str = hp.auto(S3RemoteFilesystem, 'bucket')
    prefix: str = hp.auto(S3RemoteFilesystem, 'prefix')
    region_name: Optional[str] = hp.auto(S3RemoteFilesystem, 'region_name')
    endpoint_url: Optional[str] = hp.auto(S3RemoteFilesystem, 'endpoint_url')
    # Not including the credentials as part of the hparams -- they should be specified through the default
    # environment variables
    client_config: Optional[Dict[Any, Any]] = hp.auto(S3RemoteFilesystem, 'client_config')
    transfer_config: Optional[Dict[Any, Any]] = hp.auto(S3RemoteFilesystem, 'transfer_config')

    def get_object_store_cls(self) -> Type[RemoteFilesystem]:
        return S3RemoteFilesystem

    def get_kwargs(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass
class SFTPRemoteFilesystemHparams(RemoteFilesystemHparams):
    """:class:`~.SFTPRemoteFilesystem` hyperparameters.

    The :class:`.SFTPRemoteFilesystem` uses :mod:`paramiko` to upload and download files from SSH servers

    .. note::

        To follow best security practices, credentials shouldn't be specified as part of the hyperparameters.
        Instead, please ensure that credentials are in the environment, which will be read automatically.

    Args:
        host (str): See :class:`.SFTPRemoteFilesystem`.
        port (int, optional): See :class:`.SFTPRemoteFilesystem`.
        username (str, optional): See :class:`.SFTPRemoteFilesystem`.
        known_hosts_filename (str, optional): See :class:`.SFTPRemoteFilesystem`.
        key_filename (str, optional): See :class:`.SFTPRemoteFilesystem`.
        cwd (str, optional): See :class:`.SFTPRemoteFilesystem`.
        connect_kwargs (Dict[str, Any], optional): See :class:`.SFTPRemoteFilesystem`.
    """

    host: str = hp.auto(SFTPRemoteFilesystem, 'host')
    port: int = hp.auto(SFTPRemoteFilesystem, 'port')
    username: Optional[str] = hp.auto(SFTPRemoteFilesystem, 'username')
    known_hosts_filename: Optional[str] = hp.auto(SFTPRemoteFilesystem, 'known_hosts_filename')
    known_hosts_filename_environ: str = hp.auto(SFTPRemoteFilesystem, 'known_hosts_filename_environ')
    key_filename: Optional[str] = hp.auto(SFTPRemoteFilesystem, 'key_filename')
    key_filename_environ: str = hp.auto(SFTPRemoteFilesystem, 'key_filename_environ')
    missing_host_key_policy: str = hp.auto(SFTPRemoteFilesystem, 'missing_host_key_policy')
    cwd: str = hp.auto(SFTPRemoteFilesystem, 'cwd')
    connect_kwargs: Optional[Dict[str, Any]] = hp.auto(SFTPRemoteFilesystem, 'connect_kwargs')

    def get_object_store_cls(self) -> Type[RemoteFilesystem]:
        return SFTPRemoteFilesystem

    def get_kwargs(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


object_store_registry: Dict[str, Type[RemoteFilesystemHparams]] = {
    'libcloud': LibcloudRemoteFilesystemHparams,
    's3': S3RemoteFilesystemHparams,
    'sftp': SFTPRemoteFilesystemHparams,
}
