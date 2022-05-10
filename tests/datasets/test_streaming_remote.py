import pathlib
import time

import pytest
from PIL.Image import Image
from torch.utils.data import DataLoader

from composer.datasets.ade20k import StreamingADE20k
from composer.datasets.utils import pil_image_collate


# @pytest.mark.skip()
@pytest.mark.timeout(0)
@pytest.mark.parametrize("split", ["val"])
def test_streaming_ade20k_dataset(tmpdir: pathlib.Path, split: str) -> None:
    assert split in ["train", "val"]
    expected_samples = {"train": 20206, "val": 2000}[split]

    # Paths
    # Must have valid AWS credentials in order to access this remote S3 path
    remote = f"s3://mosaicml-internal-dataset-ade20k/mds/"
    local = str(tmpdir)

    # Build StreamingDataset
    build_start = time.time()
    dataset = StreamingADE20k(remote=remote, local=local, split=split, shuffle=False)
    build_end = time.time()
    build_dur = build_end - build_start

    # Test basic iteration
    rcvd_samples = 0
    iter_start = time.time()
    for _, sample in enumerate(dataset):
        assert len(sample) == 2
        image, annotation = sample
        assert isinstance(image, Image)
        assert isinstance(annotation, Image)
        rcvd_samples += 1
    iter_end = time.time()
    iter_dur = iter_end - iter_start
    samples_per_sec = rcvd_samples / iter_dur

    # Print debug info
    print(f"build_dur={build_dur:.2f}s, iter_dur={iter_dur:.2f}, samples_per_sec={samples_per_sec:.2f}")

    # Test all samples arrived
    assert rcvd_samples == expected_samples


# @pytest.mark.skip()
@pytest.mark.timeout(0)
@pytest.mark.parametrize("split", ["val"])
def test_streaming_ade20k_dataloader(tmpdir: pathlib.Path, split: str) -> None:
    assert split in ["train", "val"]
    expected_samples = {"train": 20206, "val": 2000}[split]

    # Paths
    # Must have valid AWS credentials in order to access this remote S3 path
    remote = f"s3://mosaicml-internal-dataset-ade20k/mds/"
    local = str(tmpdir)

    # Data loading info
    shuffle = True
    batch_size = 8
    num_workers = 8
    drop_last = False
    persistent_workers = True

    # Build StreamingDataset
    ds_build_start = time.time()
    dataset = StreamingADE20k(
        remote=remote,
        local=local,
        split=split,
        shuffle=shuffle,
        batch_size=batch_size,
    )
    ds_build_end = time.time()
    ds_build_dur = ds_build_end - ds_build_start

    # Build DataLoader
    loader_build_start = time.time()
    loader = DataLoader(dataset=dataset,
                        batch_size=batch_size,
                        num_workers=num_workers,
                        drop_last=drop_last,
                        persistent_workers=persistent_workers,
                        collate_fn=pil_image_collate)
    loader_build_end = time.time()
    loader_build_dur = loader_build_end - loader_build_start

    # Print debug info
    print(f"ds_build_dur={ds_build_dur:.2f}s, loader_build_dur={loader_build_dur:.2f}s")

    for epoch in range(3):
        rcvd_samples = 0
        epoch_start = time.time()
        for _, (images, _) in enumerate(loader):
            n_samples = images.shape[0]
            rcvd_samples += n_samples
        epoch_end = time.time()
        epoch_dur = epoch_end - epoch_start
        samples_per_sec = rcvd_samples / epoch_dur
        print(f"Epoch {epoch}: epoch_dur={epoch_dur:.2f}s, samples_per_sec={samples_per_sec:.2f}")

        # Test all samples arrived
        assert rcvd_samples == expected_samples
