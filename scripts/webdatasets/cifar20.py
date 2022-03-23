from argparse import ArgumentParser, Namespace
from typing import Any, Dict, Iterable, Tuple

import numpy as np
from torchvision.datasets import CIFAR100
from wurlitzer import pipes

from composer.datasets.webdataset_utils import create_webdataset


def parse_args() -> Namespace:
    """Parse commandline arguments."""
    args = ArgumentParser()
    args.add_argument('--in_root', type=str, required=True)
    args.add_argument('--out_root', type=str, required=True)
    args.add_argument('--train_shards', type=int, default=128)
    args.add_argument('--val_shards', type=int, default=128)
    args.add_argument('--tqdm', type=int, default=1)
    return args.parse_args()


def shuffle(dataset: CIFAR100) -> Tuple[np.ndarray, np.ndarray]:
    """Numpy-convert and shuffle a CIFAR100 dataset.

    Args:
        dataset (CIFAR100): CIFAR100 dataset object.

    Returns:
        images (np.ndarray of np.uint8): Dataset images in NCHW.
        classes (np.ndarray of np.int64): Dataset classes.
    """
    indices = np.random.permutation(len(dataset))
    images = dataset.data[indices]
    classes = np.array(dataset.targets)[indices]
    return images, classes


c100_to_c20 = np.array([
    4, 1, 14, 8, 0, 6, 7, 7, 18, 3, 3, 14, 9, 18, 7, 11, 3, 9, 7, 11, 6, 11, 5, 10, 7, 6, 13, 15, 3, 15, 0, 11, 1, 10,
    12, 14, 16, 9, 11, 5, 5, 19, 8, 8, 15, 13, 14, 17, 18, 10, 16, 4, 17, 4, 2, 0, 17, 4, 18, 17, 10, 3, 2, 12, 12, 16,
    12, 1, 9, 19, 2, 10, 0, 1, 16, 12, 9, 13, 15, 13, 16, 19, 2, 4, 6, 19, 5, 5, 8, 19, 18, 1, 2, 15, 6, 0, 17, 8, 14,
    13
])


def each_sample(images: np.ndarray, classes: np.ndarray) -> Iterable[Dict[str, Any]]:
    """Generator over each dataset sample.

    Args:
        images (np.ndarray of np.uint8): Dataset images in NCHW.
        classes (np.ndarray of np.int64): Dataset classes.

    Yields:
        Sample dicts.
    """
    for idx, (img, cls) in enumerate(zip(images, classes)):
        yield {
            '__key__': f'{idx:05d}',
            'jpg': img,
            'cls': c100_to_c20[cls],
        }


def main(args: Namespace) -> None:
    """Main: create CIFAR20 webdataset.

    Args:
        args (Namespace): Commandline arguments.
    """
    with pipes():
        dataset = CIFAR100(root=args.in_root, train=True, download=True)
    images, classes = shuffle(dataset)
    create_webdataset(each_sample(images, classes), args.out_root, 'train', len(images), args.train_shards, args.tqdm)

    with pipes():
        dataset = CIFAR100(root=args.in_root, train=False, download=True)
    images, classes = shuffle(dataset)
    create_webdataset(each_sample(images, classes), args.out_root, 'val', len(images), args.val_shards, args.tqdm)


if __name__ == '__main__':
    main(parse_args())
