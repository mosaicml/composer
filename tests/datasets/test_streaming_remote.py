import pathlib
import time
from typing import Optional, Tuple

import pytest
from torch.utils.data import DataLoader

from composer.datasets.ade20k import StreamingADE20k
from composer.datasets.coco import StreamingCOCO
from composer.datasets.imagenet import StreamingImageNet1k
from composer.datasets.streaming import StreamingDataset
from composer.datasets.utils import pil_image_collate


def get_dataset(name: str, local: str, split: str, shuffle: bool,
                batch_size: Optional[int]) -> Tuple[int, StreamingDataset]:
    dataset_map = {
        "ade20k": {
            "remote": "s3://mosaicml-internal-dataset-ade20k/mds/1/",
            "num_samples": {
                "train": 20206,
                "val": 2000,
            },
            "class": StreamingADE20k,
        },
        "imagenet1k": {
            "remote": "s3://mosaicml-internal-dataset-imagenet1k/mds/1/",
            "num_samples": {
                "train": 1281167,
                "val": 50000,
            },
            "class": StreamingImageNet1k
        },
        "coco": {
            "remote": "s3://mosaicml-internal-dataset-coco/mds/1/",
            "num_samples": {
                "train": 117266,
                "val": 4952,
            },
            "class": StreamingCOCO
        },
    }
    if name not in dataset_map and split not in dataset_map[name]["num_samples"][split]:
        raise ValueError("Could not load dataset with name={name} and split={split}")

    d = dataset_map[name]
    expected_samples = d["num_samples"][split]
    remote = d["remote"]
    dataset = d["class"](remote=remote, local=local, split=split, shuffle=shuffle, batch_size=batch_size)
    return (expected_samples, dataset)


@pytest.mark.remote()
@pytest.mark.timeout(0)
@pytest.mark.parametrize("name", [
    "ade20k",
    "imagenet1k",
    "coco",
])
@pytest.mark.parametrize("split", ["val"])
def test_streaming_remote_dataset(tmpdir: pathlib.Path, name: str, split: str) -> None:

    # Build StreamingDataset
    build_start = time.time()
    expected_samples, dataset = get_dataset(name=name, local=str(tmpdir), split=split, shuffle=False, batch_size=None)
    build_end = time.time()
    build_dur = build_end - build_start
    print("Built dataset")

    # Test basic iteration
    rcvd_samples = 0
    iter_start = time.time()
    for _ in dataset:
        rcvd_samples += 1

        if (rcvd_samples % 1000 == 0):
            print(f"samples read: {rcvd_samples}")

    iter_end = time.time()
    iter_dur = iter_end - iter_start
    samples_per_sec = rcvd_samples / iter_dur

    # Print debug info
    print(f"build_dur={build_dur:.2f}s, iter_dur={iter_dur:.2f}, samples_per_sec={samples_per_sec:.2f}")

    # Test all samples arrived
    assert rcvd_samples == expected_samples


@pytest.mark.remote()
@pytest.mark.timeout(0)
@pytest.mark.parametrize("name", [
    "ade20k",
    "imagenet1k",
    "coco",
])
@pytest.mark.parametrize("split", ["val"])
def test_streaming_remote_dataloader(tmpdir: pathlib.Path, name: str, split: str) -> None:

    # Data loading info
    shuffle = True
    batch_size = 8
    num_workers = 8
    drop_last = False
    persistent_workers = True
    collate_fn = pil_image_collate if name in ["ade20k", "imagenet1k"] else None

    # Build StreamingDataset
    ds_build_start = time.time()
    expected_samples, dataset = get_dataset(name=name,
                                            local=str(tmpdir),
                                            split=split,
                                            shuffle=shuffle,
                                            batch_size=batch_size)
    ds_build_end = time.time()
    ds_build_dur = ds_build_end - ds_build_start
    print("Built dataset")

    # Build DataLoader
    loader_build_start = time.time()
    loader = DataLoader(dataset=dataset,
                        batch_size=batch_size,
                        num_workers=num_workers,
                        drop_last=drop_last,
                        persistent_workers=persistent_workers,
                        collate_fn=collate_fn)
    loader_build_end = time.time()
    loader_build_dur = loader_build_end - loader_build_start

    # Print debug info
    print(f"ds_build_dur={ds_build_dur:.2f}s, loader_build_dur={loader_build_dur:.2f}s")

    for epoch in range(3):
        rcvd_samples = 0
        epoch_start = time.time()
        for _, batch in enumerate(loader):
            n_samples = batch[0].shape[0]
            rcvd_samples += n_samples
        epoch_end = time.time()
        epoch_dur = epoch_end - epoch_start
        samples_per_sec = rcvd_samples / epoch_dur
        print(f"Epoch {epoch}: epoch_dur={epoch_dur:.2f}s, samples_per_sec={samples_per_sec:.2f}")

        # Test all samples arrived
        assert rcvd_samples == expected_samples
