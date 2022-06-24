# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import pathlib
from typing import Type

import pytest

from composer.utils.object_store import ObjectStore
from composer.utils.object_store.object_store_hparams import (ObjectStoreHparams, SFTPObjectStoreHparams,
                                                              object_store_registry)
from tests.common.hparams import assert_in_registry, construct_from_yaml
from tests.utils.object_store.object_store_settings import (get_object_store_ctx, object_store_hparams,
                                                            object_store_kwargs)


@pytest.mark.parametrize('constructor', object_store_hparams)
@pytest.mark.timeout(5)
def test_object_store_hparams_is_constructable(
    constructor: Type[ObjectStoreHparams],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
):
    yaml_dict = object_store_kwargs[constructor]
    instance = construct_from_yaml(constructor, yaml_dict=yaml_dict)
    with get_object_store_ctx(instance.get_object_store_cls(), monkeypatch, tmp_path):
        with instance.initialize_object() as object_store:
            assert isinstance(object_store, ObjectStore)


@pytest.mark.parametrize('constructor', object_store_hparams)
def test_hparams_in_registry(constructor: Type[ObjectStoreHparams]):
    assert_in_registry(constructor, object_store_registry)


def test_sftp_key_file_environ(monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path):
    private_key_filepath = '/path/to/private/key/file'
    monkeypatch.setenv('COMPOSER_SFTP_KEY_FILE', private_key_filepath)
    hparams = SFTPObjectStoreHparams(host='host',)
    assert hparams.get_kwargs()['key_filename'] == private_key_filepath
    with get_object_store_ctx(hparams.get_object_store_cls(), monkeypatch, tmp_path):
        with hparams.initialize_object() as object_store:
            assert isinstance(object_store, ObjectStore)
