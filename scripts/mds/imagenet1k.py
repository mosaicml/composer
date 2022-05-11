from argparse import ArgumentParser, Namespace
from glob import glob
import numpy as np
import os
from random import shuffle
from typing import Any, Dict, Iterable, List, Tuple

from composer.datasets.streaming import StreamingDatasetWriter


def parse_args() -> Namespace:
    """Parse commandline arguments."""
    args = ArgumentParser()
    args.add_argument('--in_root', type=str, required=True)
    args.add_argument('--out_root', type=str, required=True)
    args.add_argument('--shard_size_limit', type=int, default=1 << 25)
    args.add_argument('--tqdm', type=int, default=1)
    return args.parse_args()


def get(split_dir: str) -> List[Tuple[str, int]]:
    """Collect the samples for this dataset split.

    Args:
        split_dir (str): Input dataset split directory.

    Returns:
        List of pairs of (image filename, class ID).
    """
    pattern = os.path.join(split_dir, '*', '*.JPEG')
    filenames = sorted(glob(pattern))
    wnid2class = {}
    pairs = []
    for filename in filenames:
        parts = filename.split(os.path.sep)
        wnid = parts[-2]
        cls = wnid2class.get(wnid)
        if cls is None:
            cls = len(wnid2class)
            wnid2class[wnid] = cls
        pairs.append((filename, cls))
    shuffle(pairs)
    return pairs


def each(pairs: List[Tuple[str, int]]) -> Iterable[Dict[str, Any]]:
    """Generator over each dataset sample.

    Args:
        pairs (list): List of pairs of (image filename, class ID).

    Yields:
        Sample dicts.
    """
    for f, y in pairs:
        x = open(f, 'rb').read()
        yield {
            'x': x,
            'y': np.int64(y).tobytes(),
        }


def main(args: Namespace) -> None:
    """Main: create ImageNet1k streaming dataset.

    Args:
        args (Namespace): Commandline arguments.
    """
    fields = 'x', 'y'

    in_split_dir = os.path.join(args.in_root, 'train')
    pairs = get(in_split_dir)
    out_split_dir = os.path.join(args.out_root, 'train')
    with StreamingDatasetWriter(out_split_dir, fields, args.shard_size_limit) as out:
        out.write_samples(each(pairs), bool(args.tqdm), len(pairs))

    in_split_dir = os.path.join(args.in_root, 'val')
    pairs = get(in_split_dir)
    out_split_dir = os.path.join(args.out_root, 'val')
    with StreamingDatasetWriter(out_split_dir, fields, args.shard_size_limit) as out:
        out.write_samples(each(pairs), bool(args.tqdm), len(pairs))


if __name__ == '__main__':
    main(parse_args())
