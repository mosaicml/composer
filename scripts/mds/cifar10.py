# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import os
from argparse import ArgumentParser, Namespace
from typing import Any, Dict, Iterable, Tuple

import numpy as np
from numpy.typing import NDArray
from torchvision.datasets import CIFAR10
from wurlitzer import pipes

from composer.datasets.streaming import StreamingDatasetWriter


def parse_args() -> Namespace:
    """Parse commandline arguments."""
    args = ArgumentParser()
    args.add_argument('--in_root', type=str, required=True)
    args.add_argument('--out_root', type=str, required=True)
    args.add_argument('--shard_size_limit', type=int, default=1 << 21)
    args.add_argument('--tqdm', type=int, default=1)
    return args.parse_args()


def get(dataset: CIFAR10) -> Tuple[NDArray[np.uint8], NDArray[np.int64]]:
    """Numpy-convert and shuffle a CIFAR10 dataset.

    Args:
        dataset (CIFAR10): CIFAR10 dataset object.

    Returns:
        images (NDArray[np.uint8]): Dataset images in NCHW.
        classes (NDArray[np.int64]): Dataset classes.
    """
    indices = np.random.permutation(len(dataset))
    images = dataset.data[indices]
    classes = np.array(dataset.targets, np.int64)[indices]
    return images, classes


def each(images: NDArray[np.uint8], classes: NDArray[np.int64]) -> Iterable[Dict[str, Any]]:
    """Generator over each dataset sample.

    Args:
        images (NDArray[np.uint8]): Dataset images in NCHW.
        classes (NDArray[np.int64]): Dataset classes.

    Yields:
        Sample dicts.
    """
    for x, y in zip(images, classes):
        yield {
            'x': x.tobytes(),
            'y': y.tobytes(),
        }


def main(args: Namespace) -> None:
    """Main: create CIFAR10 streaming dataset.

    Args:
        args (Namespace): Commandline arguments.
    """
    fields = 'x', 'y'

    with pipes():
        dataset = CIFAR10(root=args.in_root, train=True, download=True)
    images, classes = get(dataset)
    split_dir = os.path.join(args.out_root, 'train')
    with StreamingDatasetWriter(split_dir, fields, args.shard_size_limit) as out:
        out.write_samples(each(images, classes), bool(args.tqdm), len(images))

    with pipes():
        dataset = CIFAR10(root=args.in_root, train=False, download=True)
    images, classes = get(dataset)
    split_dir = os.path.join(args.out_root, 'val')
    with StreamingDatasetWriter(split_dir, fields, args.shard_size_limit) as out:
        out.write_samples(each(images, classes), bool(args.tqdm), len(images))


if __name__ == '__main__':
    main(parse_args())
