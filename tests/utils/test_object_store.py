# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import os
import pathlib

import pytest

from composer.utils.object_store import ObjectStoreHparams


def test_object_store(tmpdir: pathlib.Path, monkeypatch: pytest.MonkeyPatch):
    remote_dir = str(tmpdir / "remote_dir")
    os.makedirs(remote_dir)
    local_dir = str(tmpdir / "local_dir")
    os.makedirs(local_dir)
    monkeypatch.setenv("OBJECT_STORE_KEY", remote_dir)  # for the local option, the key is the path
    provider_hparams = ObjectStoreHparams(
        provider='local',
        key_environ="OBJECT_STORE_KEY",
        container=".",
    )
    provider = provider_hparams.initialize_object()
    assert provider.provider_name == "Local Storage"
    assert provider.container_name == "."
    local_file_path = os.path.join(local_dir, "dummy_file")
    with open(local_file_path, "w+") as f:
        f.write("Hello, world!")

    provider.upload_object(local_file_path, "upload_object")
    local_file_path_download = os.path.join(local_dir, "dummy_file_downloaded")
    provider.download_object("upload_object", local_file_path_download)
    with open(local_file_path_download, "r") as f:
        assert f.read() == "Hello, world!"

    provider.upload_object_via_stream(b"Hello!", "upload_stream")
    data = provider.download_object_as_stream("upload_stream")
    assert list(data) == [b"Hello!"]
