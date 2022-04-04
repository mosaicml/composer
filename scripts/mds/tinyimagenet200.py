from argparse import ArgumentParser, Namespace
import os
import numpy as np
from PIL import Image
from random import shuffle
from tqdm import tqdm
from typing import Any, Dict, Iterable, List, Tuple
from wurlitzer import pipes

from composer.datasets.mosdataset import MosaicDatasetWriter


def parse_args() -> Namespace:
    """Parse commandline arguments."""
    args = ArgumentParser()
    args.add_argument('--in_root', type=str, required=True)
    args.add_argument('--out_root', type=str, required=True)
    args.add_argument('--shard_size_limit', type=int, default=1 << 23)
    args.add_argument('--tqdm', type=int, default=1)
    return args.parse_args()


def get_train(in_root: str, wnids: List[str]) -> List[Tuple[str, int]]:
    """Get training split sample pairs.

    Args:
        in_root (str): Input dataset root directory.
        wnids (list): Ordered list of WordNet IDs.

    Returns:
        List of pairs of (image filename, class ID).
    """
    pairs = []
    for wnid_idx, wnid in tqdm(enumerate(wnids), leave=False):
        in_dir = os.path.join(in_root, 'train', wnid, 'images')
        for basename in os.listdir(in_dir):
            filename = os.path.join(in_dir, basename)
            pairs.append((filename, wnid_idx))
    shuffle(pairs)
    return pairs


def get_val(in_root: str, wnid2idx: Dict[str, int]) -> List[Tuple[str, int]]:
    """Get validation split sample pairs.

    Args:
        in_root (str): Input dataset root directory.
        wnid2idx (dict): Mapping of WordNet ID to class ID.

    Returns:
        List of pairs of (image filename, class ID).
    """
    pairs = []
    filename = os.path.join(in_root, 'val', 'val_annotations.txt')
    lines = open(filename).read().strip().split('\n')
    for line in tqdm(lines, leave=False):
        basename, wnid = line.split()[:2]
        filename = os.path.join(in_root, 'val', 'images', basename)
        wnid_idx = wnid2idx[wnid]
        pairs.append((filename, wnid_idx))
    shuffle(pairs)
    return pairs


def each(pairs: List[Tuple[str, int]]) -> Iterable[Dict[str, Any]]:
    """Generator over each dataset sample.

    Args:
        pairs (list of (str, int)): List of pairs of (image filename, class ID).

    Yields:
        Sample dicts.
    """
    for f, y in pairs:
        x = Image.open(f)
        x = np.array(x)
        if x.ndim == 2:
            x = np.stack([x] * 3, 2)
        yield {
            'x': x.tobytes(),
            'y': np.int64(y).tobytes(),
        }


def main(args: Namespace) -> None:
    """Main: create TinyImageNet200 Mosaic dataset.

    Args:
        args (Namespace): Commandline arguments.
    """
    fields = 'x', 'y'

    filename = os.path.join(args.in_root, 'wnids.txt')
    wnids = open(filename).read().split()
    wnid2idx = dict(zip(wnids, range(len(wnids))))

    pairs = get_train(args.in_root, wnids)
    split_dir = os.path.join(args.out_root, 'train')
    with MosaicDatasetWriter(split_dir, fields, args.shard_size_limit) as out:
        out.write_samples(each(pairs), bool(args.tqdm), len(pairs))

    pairs = get_val(args.in_root, wnid2idx)
    split_dir = os.path.join(args.out_root, 'val')
    with MosaicDatasetWriter(split_dir, fields, args.shard_size_limit) as out:
        out.write_samples(each(pairs), bool(args.tqdm), len(pairs))


if __name__ == '__main__':
    main(parse_args())
