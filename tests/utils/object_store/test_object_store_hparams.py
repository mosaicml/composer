# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import pathlib
from typing import Type

import pytest

from composer.utils.object_store import ObjectStore
from composer.utils.object_store.object_store_hparams import (ObjectStoreHparams, SFTPObjectStoreHparams,
                                                              object_store_registry)
from tests.common.hparams import assert_in_registry, construct_from_yaml
from tests.utils.object_store.object_store_settings import (get_object_store_ctx, object_store_hparam_kwargs,
                                                            object_store_hparams)


@pytest.mark.parametrize('constructor', object_store_hparams)
def test_object_store_hparams_is_constructable(
    constructor: Type[ObjectStoreHparams],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
):
    yaml_dict = object_store_hparam_kwargs[constructor]
    instance = construct_from_yaml(constructor, yaml_dict=yaml_dict)
    with get_object_store_ctx(instance.get_object_store_cls(), yaml_dict, monkeypatch, tmp_path):
        with instance.initialize_object() as object_store:
            assert isinstance(object_store, ObjectStore)


@pytest.mark.parametrize('constructor', object_store_hparams)
def test_hparams_in_registry(constructor: Type[ObjectStoreHparams]):
    assert_in_registry(constructor, object_store_registry)


@pytest.mark.parametrize('kwarg_name,environ_name', [
    ['key_filename', 'COMPOSER_SFTP_KEY_FILE'],
    ['known_hosts_filename', 'COMPOSER_SFTP_KNOWN_HOSTS_FILE'],
])
@pytest.mark.filterwarnings(r'ignore:setDaemon\(\) is deprecated:DeprecationWarning')
def test_filenames_as_environs(monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path, kwarg_name: str,
                               environ_name: str):
    key_filepath = str(tmp_path / 'keyfile')
    with open(key_filepath, 'w+') as f:
        f.write('')

    monkeypatch.setenv(environ_name, key_filepath)
    hparams = SFTPObjectStoreHparams(host='host',)
    yaml_dict = object_store_hparam_kwargs[SFTPObjectStoreHparams]
    assert hparams.get_kwargs()[kwarg_name] == key_filepath
    with get_object_store_ctx(hparams.get_object_store_cls(), yaml_dict, monkeypatch, tmp_path):
        with hparams.initialize_object() as object_store:
            assert isinstance(object_store, ObjectStore)
