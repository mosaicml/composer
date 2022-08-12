# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""CIFAR10 streaming dataset conversion scripts."""

import os
from argparse import ArgumentParser, Namespace
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
from numpy.typing import NDArray
from torchvision.datasets import CIFAR10

from composer.datasets.streaming import StreamingDatasetWriter


def parse_args() -> Namespace:
    """Parse commandline arguments."""
    args = ArgumentParser()
    args.add_argument('--in_root', type=str, required=True)
    args.add_argument('--out_root', type=str, required=True)
    args.add_argument('--shard_size_limit', type=int, default=1 << 21)
    args.add_argument('--tqdm', type=int, default=1)
    return args.parse_args()


def get(dataset: CIFAR10, shuffle: bool) -> Tuple[NDArray[np.int64], NDArray[np.uint8], NDArray[np.int64]]:
    """Numpy-convert and shuffle a CIFAR10 dataset.

    Args:
        dataset (CIFAR10): CIFAR10 dataset object.

    Returns:
        uids (NDArray[np.int64]): Sample uids.
        images (NDArray[np.uint8]): Dataset images in NCHW.
        classes (NDArray[np.int64]): Dataset classes.
    """
    uids = np.arange(len(dataset), dtype=np.int64)
    images = dataset.data
    classes = np.array(dataset.targets, np.int64)
    if shuffle:
        perm = np.random.permutation(len(dataset))
        uids = uids[perm]
        images = images[perm]
        classes = classes[perm]
    return uids, images, classes


def each(uids: NDArray[np.int64], images: NDArray[np.uint8], classes: NDArray[np.int64]) -> Iterable[Dict[str, Any]]:
    """Generator over each dataset sample.

    Args:
        uids (NDArray[np.int64]): Dataset uids.
        images (NDArray[np.uint8]): Dataset images in NCHW.
        classes (NDArray[np.int64]): Dataset classes.

    Yields:
        Sample dicts.
    """
    for uid, x, y in zip(uids, images, classes):
        yield {
            'uid': uid.tobytes(),
            'x': x.tobytes(),
            'y': y.tobytes(),
        }


def main(args: Namespace) -> None:
    """Main: create CIFAR10 streaming dataset.

    Args:
        args (Namespace): Commandline arguments.
    """
    fields = ['uid', 'x', 'y']

    for (split, expected_num_samples, shuffle) in [
        ('train', 50000, True),
        ('val', 10000, False),
    ]:
        dataset = CIFAR10(root=args.in_root, train=(split == 'train'), download=True)
        uids, images, classes = get(dataset=dataset, shuffle=shuffle)
        assert len(images) == expected_num_samples

        split_dir = os.path.join(args.out_root, split)
        with StreamingDatasetWriter(split_dir, fields, args.shard_size_limit) as out:
            out.write_samples(each(uids, images, classes), bool(args.tqdm), expected_num_samples)


if __name__ == '__main__':
    main(parse_args())
