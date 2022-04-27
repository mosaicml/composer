import math
import os
import shutil
import tempfile
import time
from typing import Dict, Iterable, List, Optional

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from composer.datasets.streaming import StreamingDataset, StreamingDatasetWriter


def get_fake_samples_decoders(num_samples: int) -> List[Dict[str, bytes]]:
    samples = [{"uid": f"{ix:06}".encode("utf-8"), "data": (3 * ix).to_bytes(4, "big")} for ix in range(num_samples)]
    decoders = {
        "uid": lambda uid_bytes: uid_bytes.decode("utf-8"),
        "data": lambda data_bytes: int.from_bytes(data_bytes, "big")
    }
    return samples, decoders


def write_synthetic_streaming_dataset(samples: List[Dict[str, bytes]], shard_size_limit: int) -> str:
    tmpdir = tempfile.TemporaryDirectory().name
    first_sample_fields = list(samples[0].keys())
    with StreamingDatasetWriter(dirname=tmpdir, fields=first_sample_fields,
                                shard_size_limit=shard_size_limit) as writer:
        writer.write_samples(samples=samples)
    return tmpdir


@pytest.mark.parametrize("num_samples", [100, 10000])
@pytest.mark.parametrize("shard_size_limit", [1 << 8, 1 << 16, 1 << 24])
def test_writer(num_samples: int, shard_size_limit: int) -> None:
    samples, _ = get_fake_samples_decoders(num_samples)

    first_sample_values = samples[0].values()
    first_sample_byte_sizes = np.array([len(v) for v in first_sample_values], dtype=np.int64)
    first_sample_bytes = len(first_sample_byte_sizes.tobytes() + b''.join(first_sample_values))

    expected_samples_per_shard = shard_size_limit // first_sample_bytes
    expected_num_shards = math.ceil(num_samples / expected_samples_per_shard)
    expected_num_files = expected_num_shards + 1  # the index file

    dirname = write_synthetic_streaming_dataset(samples=samples, shard_size_limit=shard_size_limit)
    files = os.listdir(dirname)

    assert len(files) == expected_num_files, f"Files written ({len(files)}) != expected ({expected_num_files})."


@pytest.mark.parametrize("share_remote_local", [False, True])
@pytest.mark.parametrize("shuffle", [False, True])
def test_reader(share_remote_local: bool, shuffle: bool):
    num_samples = 117
    shard_size_limit = 1 << 8
    samples, decoders = get_fake_samples_decoders(num_samples)
    remote = write_synthetic_streaming_dataset(samples=samples, shard_size_limit=shard_size_limit)
    if share_remote_local:
        local = remote
    else:
        local = tempfile.TemporaryDirectory().name

    # Build StreamingDataset
    dataset = StreamingDataset(remote=remote, local=local, shuffle=shuffle, decoders=decoders)

    # Test length
    assert len(dataset) == num_samples, f"Got dataset length={len(dataset)} samples, expected {num_samples}"

    # Test basic sample order
    shuffle_matches = 0
    for ix, sample in enumerate(dataset):
        uid = sample["uid"]
        data = sample["data"]
        expected_uid = f"{ix:06}"
        expected_data = 3 * ix
        if shuffle:
            shuffle_matches += (expected_uid == uid)
        else:
            assert uid == expected_uid == uid, f"sample ix={ix} has uid={uid}, expected {expected_uid}"
            assert data == expected_data, f"sample ix={ix} has data={data}, expected {expected_data}"

    # If shuffling, there should be few matches
    # The probability of k matches in a random permutation is ~1/(e*(k!))
    if shuffle:
        assert shuffle_matches < 10


@pytest.mark.timeout(10)
@pytest.mark.parametrize("created_ago", [0.5, 3])
@pytest.mark.parametrize("timeout", [1])
def test_reader_after_crash(created_ago, timeout):
    num_samples = 117
    shard_size_limit = 1 << 8
    samples, decoders = get_fake_samples_decoders(num_samples)
    remote = write_synthetic_streaming_dataset(samples=samples, shard_size_limit=shard_size_limit)
    local = tempfile.TemporaryDirectory().name

    os.makedirs(local, exist_ok=True)
    shutil.copy(os.path.join(remote, "index.mds"), os.path.join(local, "index.mds.tmp"))
    shutil.copy(os.path.join(remote, "000003.mds"), os.path.join(local, "000003.mds.tmp"))
    time.sleep(created_ago)
    dataset = StreamingDataset(remote=remote, local=local, shuffle=False, decoders=decoders, timeout=timeout)

    # Iterate over dataset and make sure there are no TimeoutErrors
    for ix, sample in enumerate(dataset):
        pass


@pytest.mark.parametrize(
    "share_remote_local",
    [
        True,
        pytest.param(False, marks=pytest.mark.xfail(reason="__getitem__ currently expects shards to exist")),
    ],
)
def test_reader_getitem(share_remote_local: bool):
    num_samples = 117
    shard_size_limit = 1 << 8
    samples, decoders = get_fake_samples_decoders(num_samples)
    remote = write_synthetic_streaming_dataset(samples=samples, shard_size_limit=shard_size_limit)
    if share_remote_local:
        local = remote
    else:
        local = tempfile.TemporaryDirectory().name

    # Build StreamingDataset
    dataset = StreamingDataset(remote=remote, local=local, shuffle=False, decoders=decoders)

    # Test retrieving random sample
    try:
        sample = dataset[17]
    except Exception as e:
        assert False, f"Unable to get random sample, got exception: {e}"


@pytest.mark.parametrize("batch_size", [1, 2, 5, 128])
@pytest.mark.parametrize("drop_last", [False, True])
@pytest.mark.parametrize("num_workers", [0, 1, 8])
@pytest.mark.parametrize("share_remote_local", [False, True])
@pytest.mark.parametrize("shuffle", [False, True])
def test_dataloader_single_device(batch_size: int, drop_last: bool, num_workers: int, share_remote_local: bool,
                                  shuffle: bool):
    num_samples = 1234
    shard_size_limit = 1 << 8
    samples, decoders = get_fake_samples_decoders(num_samples)
    remote = write_synthetic_streaming_dataset(samples=samples, shard_size_limit=shard_size_limit)
    if share_remote_local:
        local = remote
    else:
        local = tempfile.TemporaryDirectory().name

    # Build StreamingDataset
    dataset = StreamingDataset(remote=remote, local=local, shuffle=shuffle, decoders=decoders, batch_size=batch_size)

    # Build DataLoader
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, num_workers=num_workers, drop_last=drop_last)

    # Test dataloader length
    expected_num_batches = (num_samples // batch_size) if drop_last else math.ceil(num_samples / batch_size)
    num_batches = len(dataloader)
    assert num_batches == expected_num_batches, f"Got dataloader length={num_batches} batches, expected {expected_num_batches}"
