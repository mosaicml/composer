# Copyright 2021 MosaicML. All Rights Reserved.

import os
import pathlib

import pytest

from composer.utils.file_retriever import GetFileNotFoundException, get_file
from composer.utils.object_store import ObjectStoreHparams


@pytest.mark.xfail(reason="Occassionally hits the timeout. Should refactor to use a local webserver.")
def test_get_file_uri(tmpdir: pathlib.Path):
    get_file(
        path="https://www.mosaicml.com",
        object_store=None,
        destination=str(tmpdir / "example"),
        chunk_size=1024 * 1024,
        progress_bar=False,
    )
    with open(str(tmpdir / "example"), "r") as f:
        assert f.readline().startswith("<!")


@pytest.mark.xfail(reason="Occassionally hits the timeout. Should refactor to use a local webserver.")
def test_get_file_uri_not_found(tmpdir: pathlib.Path):
    with pytest.raises(GetFileNotFoundException):
        get_file(
            path="https://www.mosaicml.com/notfounasdfjilasdfjlkasdljkasjdklfljkasdjfk",
            object_store=None,
            destination=str(tmpdir / "example"),
            chunk_size=1024 * 1024,
            progress_bar=False,
        )


def test_get_file_object_store(tmpdir: pathlib.Path, monkeypatch: pytest.MonkeyPatch):
    remote_dir = tmpdir / "remote_dir"
    os.makedirs(remote_dir)
    monkeypatch.setenv("OBJECT_STORE_KEY", str(remote_dir))  # for the local option, the key is the path
    provider = ObjectStoreHparams(
        provider='local',
        key_environ="OBJECT_STORE_KEY",
        container=".",
    ).initialize_object()
    with open(str(remote_dir / "checkpoint.txt"), 'wb') as f:
        f.write(b"checkpoint1")
    get_file(path="checkpoint.txt",
             object_store=provider,
             destination=str(tmpdir / "example"),
             chunk_size=1024 * 1024,
             progress_bar=False)
    with open(str(tmpdir / "example"), "rb") as f:
        f.read() == b"checkpoint1"


def test_get_file_object_store_not_found(tmpdir: pathlib.Path, monkeypatch: pytest.MonkeyPatch):
    remote_dir = tmpdir / "remote_dir"
    os.makedirs(remote_dir)
    monkeypatch.setenv("OBJECT_STORE_KEY", str(remote_dir))  # for the local option, the key is the path
    provider = ObjectStoreHparams(
        provider='local',
        key_environ="OBJECT_STORE_KEY",
        container=".",
    ).initialize_object()
    with pytest.raises(GetFileNotFoundException):
        get_file(path="checkpoint.txt",
                 object_store=provider,
                 destination=str(tmpdir / "example"),
                 chunk_size=1024 * 1024,
                 progress_bar=False)


def test_get_file_local_path(tmpdir: pathlib.Path):
    tmpfile_name = os.path.join(tmpdir, "file.txt")
    with open(tmpfile_name, "x") as f:
        f.write("hi!")

    get_file(
        path=tmpfile_name,
        object_store=None,
        destination=str(tmpdir / "example"),
        chunk_size=1024 * 1024,
        progress_bar=False,
    )
    with open(str(tmpdir / "example"), "r") as f:
        assert f.read() == "hi!"


def test_get_file_local_path_not_found():
    with pytest.raises(GetFileNotFoundException):
        get_file(
            path="/path/does/not/exist",
            object_store=None,
            destination="destination",
            chunk_size=1024 * 1024,
            progress_bar=False,
        )
