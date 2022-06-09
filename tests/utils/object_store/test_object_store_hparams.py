# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

from typing import Type

import pytest

from composer.utils.object_store.object_store import ObjectStore
from composer.utils.object_store.object_store_hparams import (LibcloudObjectStoreHparams, ObjectStoreHparams,
                                                              object_store_registry)
from tests.common.hparams import assert_in_registry, construct_from_yaml
from tests.utils.object_store.object_store_settings import object_store_hparams, object_store_kwargs


@pytest.mark.parametrize('constructor', object_store_hparams)
@pytest.mark.timeout(5)
def test_object_store_hparams_is_constructable(
    constructor: Type[ObjectStoreHparams],
    monkeypatch: pytest.MonkeyPatch,
):
    # The ObjectStoreLogger needs the OBJECT_STORE_KEY set
    yaml_dict = object_store_kwargs[constructor]
    if constructor is LibcloudObjectStoreHparams:
        monkeypatch.setenv('OBJECT_STORE_KEY', '.')
    instance = construct_from_yaml(constructor, yaml_dict=yaml_dict)
    object_store = instance.initialize_object()
    assert isinstance(object_store, ObjectStore)


@pytest.mark.parametrize('constructor', object_store_hparams)
def test_hparams_in_registry(constructor: Type[ObjectStoreHparams]):
    assert_in_registry(constructor, object_store_registry)
