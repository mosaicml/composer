# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict, Type, Union

import pytest

import composer.utils.object_store
import composer.utils.object_store.object_store_hparams
from composer.utils.object_store import LibcloudObjectStore, ObjectStore
from composer.utils.object_store.object_store_hparams import LibcloudObjectStoreHparams, ObjectStoreHparams
from tests.common import get_module_subclasses

try:
    import libcloud
    _LIBCLOUD_AVAILABLE = True
    del libcloud
except ImportError:
    _LIBCLOUD_AVAILABLE = False

_object_store_marks = {
    LibcloudObjectStore: [pytest.mark.skipif(not _LIBCLOUD_AVAILABLE, reason="Missing dependency")],
    LibcloudObjectStoreHparams: [pytest.mark.skipif(not _LIBCLOUD_AVAILABLE, reason="Missing dependency")],
}

object_store_kwargs: Dict[Union[Type[ObjectStore], Type[ObjectStoreHparams]], Dict[str, Any]] = {
    LibcloudObjectStore: {
        'provider': 'local',
        'container': '.',
        'provider_kwargs': {
            'key': '.',
        },
    },
    LibcloudObjectStoreHparams: {
        "provider": 'local',
        "key_environ": "OBJECT_STORE_KEY",
        "container": ".",
    }
}

object_stores = [
    pytest.param(x, marks=_object_store_marks[x], id=x.__name__)
    for x in get_module_subclasses(composer.utils.object_store, ObjectStore)
]
object_store_hparams = [
    pytest.param(x, marks=_object_store_marks[x], id=x.__name__)
    for x in get_module_subclasses(composer.utils.object_store.object_store_hparams, ObjectStoreHparams)
]
