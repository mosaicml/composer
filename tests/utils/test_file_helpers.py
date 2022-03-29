# Copyright 2021 MosaicML. All Rights Reserved.

import os
import pathlib

import pytest

from composer.core.time import Time, Timestamp
from composer.utils.file_helpers import (GetFileNotFoundException, ensure_folder_is_empty, format_name_with_dist,
                                         format_name_with_dist_and_time, get_file, is_tar)
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


def test_is_tar():
    assert is_tar("x.tar")
    assert is_tar("x.tgz")
    assert is_tar("x.tar.gz")
    assert is_tar("x.tar.bz2")
    assert is_tar("x.tar.lzma")
    assert not is_tar("x")


def test_format_name_with_dist():
    vars = ["run_name", "rank", "node_rank", "world_size", "local_world_size", "local_rank", "extra"]
    format_str = ','.join(f"{x}={{{x}}}" for x in vars)
    expected_str = "run_name=awesome_run,rank=0,node_rank=0,world_size=1,local_world_size=1,local_rank=0,extra=42"
    assert format_name_with_dist(format_str, "awesome_run", extra=42) == expected_str


def test_format_name_with_dist_and_time():
    vars = [
        "run_name",
        "rank",
        "node_rank",
        "world_size",
        "local_world_size",
        "local_rank",
        "extra",
        "epoch",
        "batch",
        "batch_in_epoch",
        "sample",
        "sample_in_epoch",
        "token",
        "token_in_epoch",
    ]
    format_str = ','.join(f"{x}={{{x}}}" for x in vars)
    expected_str = ("run_name=awesome_run,rank=0,node_rank=0,world_size=1,local_world_size=1,local_rank=0,extra=42,"
                    "epoch=0,batch=1,batch_in_epoch=1,sample=2,sample_in_epoch=2,token=3,token_in_epoch=3")
    timestamp = Timestamp(
        epoch=Time.from_timestring("0ep"),
        batch=Time.from_timestring("1ba"),
        batch_in_epoch=Time.from_timestring("1ba"),
        sample=Time.from_timestring("2sp"),
        sample_in_epoch=Time.from_timestring("2sp"),
        token=Time.from_timestring("3tok"),
        token_in_epoch=Time.from_timestring("3tok"),
    )
    assert format_name_with_dist_and_time(format_str, "awesome_run", timestamp=timestamp, extra=42) == expected_str


def test_ensure_folder_is_empty(tmpdir: pathlib.Path):
    ensure_folder_is_empty(tmpdir)
