from argparse import ArgumentParser, Namespace
import os
import numpy as np
from torchvision.datasets import CIFAR100
from typing import Any, Dict, Iterable, Tuple
from wurlitzer import pipes

from composer.datasets.mosdataset import MosaicDatasetWriter


def parse_args() -> Namespace:
    """Parse commandline arguments."""
    args = ArgumentParser()
    args.add_argument('--in_root', type=str, required=True)
    args.add_argument('--out_root', type=str, required=True)
    args.add_argument('--shard_size_limit', type=int, default=1 << 21)
    args.add_argument('--tqdm', type=int, default=1)
    return args.parse_args()


def get(dataset: CIFAR100) -> Tuple[np.ndarray, np.ndarray]:
    """Numpy-convert and shuffle a CIFAR100 dataset.

    Args:
        dataset (CIFAR100): CIFAR100 dataset object.

    Returns:
        images (np.ndarray of np.uint8): Dataset images in NCHW.
        classes (np.ndarray of np.int64): Dataset classes.
    """
    indices = np.random.permutation(len(dataset))
    images = dataset.data[indices]
    classes = np.array(dataset.targets, np.int64)[indices]
    return images, classes


cifar100_to_cifar20 = np.array([
    4, 1, 14, 8, 0, 6, 7, 7, 18, 3, 3, 14, 9, 18, 7, 11, 3, 9, 7, 11, 6, 11, 5, 10, 7, 6, 13, 15, 3, 15, 0, 11, 1, 10,
    12, 14, 16, 9, 11, 5, 5, 19, 8, 8, 15, 13, 14, 17, 18, 10, 16, 4, 17, 4, 2, 0, 17, 4, 18, 17, 10, 3, 2, 12, 12, 16,
    12, 1, 9, 19, 2, 10, 0, 1, 16, 12, 9, 13, 15, 13, 16, 19, 2, 4, 6, 19, 5, 5, 8, 19, 18, 1, 2, 15, 6, 0, 17, 8, 14,
    13
], np.int64)


def each(images: np.ndarray, classes: np.ndarray) -> Iterable[Dict[str, Any]]:
    """Generator over each dataset sample.

    Args:
        images (np.ndarray of np.uint8): Dataset images in NCHW.
        classes (np.ndarray of np.int64): Dataset classes.

    Yields:
        Sample dicts.
    """
    for x, y in zip(images, classes):
        y = cifar100_to_cifar20[y]
        yield {
            'x': x.tobytes(),
            'y': y.tobytes(),
        }


def main(args: Namespace) -> None:
    """Main: create CIFAR100 Mosaic dataset.

    Args:
        args (Namespace): Commandline arguments.
    """
    fields = 'x', 'y'

    with pipes():
        dataset = CIFAR100(root=args.in_root, train=True, download=True)
    images, classes = get(dataset)
    split_dir = os.path.join(args.out_root, 'train')
    with MosaicDatasetWriter(split_dir, fields, args.shard_size_limit) as out:
        out.write_samples(each(images, classes), bool(args.tqdm), len(images))

    with pipes():
        dataset = CIFAR100(root=args.in_root, train=False, download=True)
    images, classes = get(dataset)
    split_dir = os.path.join(args.out_root, 'val')
    with MosaicDatasetWriter(split_dir, fields, args.shard_size_limit) as out:
        out.write_samples(each(images, classes), bool(args.tqdm), len(images))


if __name__ == '__main__':
    main(parse_args())
