from argparse import ArgumentParser, Namespace
from typing import Any, Dict, Generator, Tuple

import numpy as np
from torchvision.datasets import MNIST

from composer.datasets.webdataset import create_webdataset


def parse_args() -> Namespace:
    args = ArgumentParser()
    args.add_argument('--out_root', type=str, required=True)
    args.add_argument('--train_shards', type=int, default=128)
    args.add_argument('--val_shards', type=int, default=128)
    args.add_argument('--tqdm', type=int, default=1)
    return args.parse_args()


def shuffle(dataset: MNIST) -> Tuple[np.ndarray, np.ndarray]:
    indices = np.random.permutation(len(dataset))
    images = dataset.data[indices].numpy()
    classes = dataset.targets[indices].numpy()
    return images, classes


def each_sample(images: np.ndarray, classes: np.ndarray) -> Generator[Dict[str, Any], None, None]:
    for idx, (img, cls) in enumerate(zip(images, classes)):
        yield {
            '__key__': f'{idx:05d}',
            'jpg': img,
            'cls': cls,
        }


def main(args: Namespace) -> None:
    dataset = MNIST(root='/datasets/mnist', train=True, download=True)
    images, classes = shuffle(dataset)
    create_webdataset(each_sample(images, classes), args.out_root, 'train', len(images), args.train_shards, args.tqdm)

    dataset = MNIST(root='/datasets/mnist', train=False, download=True)
    images, classes = shuffle(dataset)
    create_webdataset(each_sample(images, classes), args.out_root, 'val', len(images), args.val_shards, args.tqdm)


if __name__ == '__main__':
    main(parse_args())
