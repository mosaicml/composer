import pathlib
import time
from typing import Tuple

import pytest
from PIL.Image import Image
from torch.utils.data import DataLoader

from composer.datasets.ade20k import StreamingADE20k
from composer.datasets.imagenet import StreamingImageNet1k
from composer.datasets.streaming import StreamingDataset
from composer.datasets.utils import pil_image_collate


def get_dataset(name: str, split: str, local: str) -> Tuple[int, StreamingDataset]:
    dataset_map = {
        "ade20k": {
            "train": (20206, lambda local: StreamingADE20k(
                remote="s3://mosaicml-internal-dataset-ade20k/mds/", local=local, split="train", shuffle=True)),
            "val": (2000, lambda local: StreamingADE20k(
                remote="s3://mosaicml-internal-dataset-ade20k/mds/", local=local, split="val", shuffle=False)),
        },
        "imagenet1k": {
            "train": (1281167,
                      lambda local: StreamingImageNet1k(remote="s3://mosaicml-internal-dataset-imagenet1k/mds-new-128/",
                                                        local=local,
                                                        split="train",
                                                        shuffle=True)),
            "val": (50000,
                    lambda local: StreamingImageNet1k(remote="s3://mosaicml-internal-dataset-imagenet1k/mds-new-128/",
                                                      local=local,
                                                      split="val",
                                                      shuffle=False)),
        }
    }
    if name not in dataset_map and split not in dataset_map[name]:
        raise ValueError("Could not load dataset with name={name} and split={split}")

    expected_samples, build_dataset_fn = dataset_map[name][split]
    dataset = build_dataset_fn(local)
    return (expected_samples, dataset)


@pytest.mark.remote()
@pytest.mark.timeout(0)
@pytest.mark.parametrize("name", ["ade20k", "imagenet1k"])
@pytest.mark.parametrize("split", ["val"])
def test_streaming_remote_dataset(tmpdir: pathlib.Path, name: str, split: str) -> None:

    # Build StreamingDataset
    build_start = time.time()
    expected_samples, dataset = get_dataset(name=name, split=split, local=str(tmpdir))
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


# @pytest.mark.remote()
# @pytest.mark.timeout(0)
# @pytest.mark.parametrize("split", ["val"])
# def test_streaming_ade20k_dataloader(tmpdir: pathlib.Path, split: str) -> None:
#     assert split in ["train", "val"]
#     expected_samples = {"train": 20206, "val": 2000}[split]

#     # Paths
#     # Must have valid AWS credentials in order to access this remote S3 path
#     remote = f"s3://mosaicml-internal-dataset-ade20k/mds/"
#     local = str(tmpdir)

#     # Data loading info
#     shuffle = True
#     batch_size = 8
#     num_workers = 8
#     drop_last = False
#     persistent_workers = True

#     # Build StreamingDataset
#     ds_build_start = time.time()
#     dataset = StreamingADE20k(
#         remote=remote,
#         local=local,
#         split=split,
#         shuffle=shuffle,
#         batch_size=batch_size,
#     )
#     ds_build_end = time.time()
#     ds_build_dur = ds_build_end - ds_build_start

#     # Build DataLoader
#     loader_build_start = time.time()
#     loader = DataLoader(dataset=dataset,
#                         batch_size=batch_size,
#                         num_workers=num_workers,
#                         drop_last=drop_last,
#                         persistent_workers=persistent_workers,
#                         collate_fn=pil_image_collate)
#     loader_build_end = time.time()
#     loader_build_dur = loader_build_end - loader_build_start

#     # Print debug info
#     print(f"ds_build_dur={ds_build_dur:.2f}s, loader_build_dur={loader_build_dur:.2f}s")

#     for epoch in range(3):
#         rcvd_samples = 0
#         epoch_start = time.time()
#         for _, (images, _) in enumerate(loader):
#             n_samples = images.shape[0]
#             rcvd_samples += n_samples
#         epoch_end = time.time()
#         epoch_dur = epoch_end - epoch_start
#         samples_per_sec = rcvd_samples / epoch_dur
#         print(f"Epoch {epoch}: epoch_dur={epoch_dur:.2f}s, samples_per_sec={samples_per_sec:.2f}")

#         # Test all samples arrived
#         assert rcvd_samples == expected_samples
