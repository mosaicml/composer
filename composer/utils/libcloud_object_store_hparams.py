# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import os
from typing import Any, Dict, Optional

import yahp as hp

from composer.utils.libcloud_object_store import LibcloudObjectStore


@dataclasses.dataclass
class LibcloudObjectStoreHparams(hp.Hparams):
    """:class:`~composer.utils.libcloud_object_store.LibcloudObjectStore` hyperparameters.

    .. rubric:: Example

    Here's an example on how to connect to an Amazon S3 bucket. This example assumes:

    * The container is named named ``MY_CONTAINER``.
    * The AWS Access Key ID is stored in an environment variable named ``AWS_ACCESS_KEY_ID``.
    * The Secret Access Key is in an environmental variable named ``AWS_SECRET_ACCESS_KEY``.

    .. testsetup:: composer.utils.libcloud_object_store.LibcloudObjectStoreHparams.__init__.s3

        import os

        os.environ["AWS_ACCESS_KEY_ID"] = "key"
        os.environ["AWS_SECRET_ACCESS_KEY"] = "secret"

    .. doctest:: composer.utils.libcloud_object_store.LibcloudObjectStoreHparams.__init__.s3

        >>> from composer.utils import LibcloudObjectStoreHparams
        >>> provider_hparams = LibcloudObjectStoreHparams(
        ...     provider="s3",
        ...     container="MY_CONTAINER",
        ...     key_environ="AWS_ACCESS_KEY_ID",
        ...     secret_environ="AWS_SECRET_ACCESS_KEY",
        ... )
        >>> provider = provider_hparams.initialize_object()
        >>> provider
        <composer.utils.libcloud_object_store.LibcloudObjectStore object at ...>

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

            .. testsetup::  composer.utils.libcloud_object_store.LibcloudObjectStoreHparams.__init__.key

                import os
                import functools
                from composer.utils import LibcloudObjectStoreHparams

                os.environ["OBJECT_STORE_KEY"] = "MY_KEY"
                LibcloudObjectStoreHparams = functools.partial(LibcloudObjectStoreHparams, provider="s3", container="container")

            .. doctest:: composer.utils.libcloud_object_store.LibcloudObjectStoreHparams.__init__.key

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

            .. testsetup:: composer.utils.libcloud_object_store.LibcloudObjectStoreHparams.__init__.secret

                import os
                import functools
                from composer.utils import LibcloudObjectStoreHparams

                original_secret = os.environ.get("OBJECT_STORE_SECRET")
                os.environ["OBJECT_STORE_SECRET"] = "MY_SECRET"
                LibcloudObjectStoreHparams = functools.partial(LibcloudObjectStoreHparams, provider="s3", container="container")


            .. doctest:: composer.utils.libcloud_object_store.LibcloudObjectStoreHparams.__init__.secret

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

    provider: str = hp.auto(LibcloudObjectStore, "provider")
    container: str = hp.auto(LibcloudObjectStore, "container")
    key_environ: Optional[str] = hp.optional(("The name of an environment variable containing "
                                              "an API key or username to use to connect to the provider."),
                                             default=None)
    secret_environ: Optional[str] = hp.optional(("The name of an environment variable containing "
                                                 "an API secret or password to use to connect to the provider."),
                                                default=None)
    region: Optional[str] = hp.optional("Cloud region to use", default=None)
    host: Optional[str] = hp.optional("Override hostname for connections", default=None)
    port: Optional[int] = hp.optional("Override port for connections", default=None)
    extra_init_kwargs: Dict[str, Any] = hp.optional(
        "Extra keyword arguments to pass into the constructor for the specified provider.", default_factory=dict)

    def get_provider_kwargs(self) -> Dict[str, Any]:
        """Returns the ``provider_kwargs`` argument, which is used to construct a :class:`.LibcloudObjectStore`.

        Returns:
            Dict[str, Any]: The ``provider_kwargs`` for use in constructing an :class:`.LibcloudObjectStore`.
        """
        init_kwargs = {}
        for key in ("host", "port", "region"):
            kwarg = getattr(self, key)
            if getattr(self, key) is not None:
                init_kwargs[key] = kwarg
        init_kwargs["key"] = None if self.key_environ is None else os.environ[self.key_environ]
        init_kwargs["secret"] = None if self.secret_environ is None else os.environ[self.secret_environ]
        init_kwargs.update(self.extra_init_kwargs)
        return init_kwargs

    def initialize_object(self):
        """Returns an instance of :class:`.LibcloudObjectStore`.

        Returns:
            LibcloudObjectStore: The object_store.
        """

        return LibcloudObjectStore(
            provider=self.provider,
            container=self.container,
            provider_kwargs=self.get_provider_kwargs(),
        )
