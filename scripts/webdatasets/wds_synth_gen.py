from typing import Any, Dict, Iterable, Tuple
from argparse import ArgumentParser, Namespace

import numpy as np

from composer.datasets.webdataset_utils import create_webdataset


def parse_args() -> Namespace:
    """Parse commandline arguments.

    Returns:
        Namespace: Commandline arguments.
    """
    args = ArgumentParser()
    args.add_argument('--samples', type=int, required=True)
    args.add_argument('--out_root', type=str, required=True)
    args.add_argument('--shards', type=int, default=64)
    args.add_argument('--splits', type=str, default='train,val')
    return args.parse_args()


def each_sample(n_samples: int) -> Iterable[Dict[str, Any]]:
    """Generate samples of the synthetic dataset.

    Args:
        n_samples (int): Number of samples, and also sample dimensionality.

    Yields:
        dict: Each sample.
    """
    for i in range(n_samples):
        x = np.zeros(n_samples, np.float32)
        x[i] = 1
        yield {
            '__key__': str(i),
            'x': x.tobytes(),
            'y': x.tobytes(),
        }


def main(args) -> None:
    """Main: create a synthetic webdataset to test sample coverage given balanced or unbalanced shards.

    Args:
        args (Namespace): Commandline arguments.
    """
    splits = args.splits.split(',')
    for split in splits:
        create_webdataset(each_sample(args.samples), args.out_root, split, args.samples, args.shards, False)


if __name__ == '__main__':
    main(parse_args())
