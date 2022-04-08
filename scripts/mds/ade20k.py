import os
from argparse import ArgumentParser, Namespace
from glob import glob
from random import shuffle
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

from composer.datasets.streaming import StreamingDatasetWriter


def parse_args() -> Namespace:
    """Parse commandline arguments."""
    args = ArgumentParser()
    args.add_argument('--in_root', type=str, required=True)
    args.add_argument('--out_root', type=str, required=True)
    args.add_argument('--shard_size_limit', type=int, default=1 << 25)
    args.add_argument('--tqdm', type=int, default=1)
    return args.parse_args()


def get(in_root: str, split: str) -> List[Tuple[str, int]]:
    """Collect the samples for this dataset split.

    Args:
        in_root (str): Input dataset directory.
        split (str): Split name.

    Returns:
        List of pairs of (image filename, annotation filename).
    """

    # Get uids
    image_glob_pattern = f'{in_root}/images/{split}/ADE_{split}_*.jpg'
    images = sorted(glob(image_glob_pattern))
    uids = [s.strip(".jpg")[-8:] for s in images]

    # Remove corrupted uids
    corrupted_uids = ['00003020', '00001701', '00013508', '00008455']
    uids = [uid for uid in uids if uid not in corrupted_uids]

    # Create and shuffle pairs
    pairs = [(f'{in_root}/images/{split}/ADE_{split}_{uid}.jpg', f'{in_root}/annotations/{split}/ADE_{split}_{uid}.png')
             for uid in uids]
    shuffle(pairs)
    return pairs


def each(pairs: List[Tuple[str, int]]) -> Iterable[Dict[str, Any]]:
    """Generator over each dataset sample.

    Args:
        pairs (list): List of pairs of (image filename, annotation filename).

    Yields:
        Sample dicts.
    """
    for idx, (image_file, annotation_file) in enumerate(pairs):
        image = open(image_file, 'rb').read()
        annotation = open(annotation_file, 'rb').read()
        yield {
            'image': image,
            'annotation': annotation,
        }


def main(args: Namespace) -> None:
    """Main: create ADE20K streaming dataset.

    Args:
        args (Namespace): Commandline arguments.
    """
    fields = 'image', 'annotation'

    pairs = get(args.in_root, 'train')
    out_split_dir = os.path.join(args.out_root, 'train')
    with StreamingDatasetWriter(out_split_dir, fields, args.shard_size_limit) as out:
        out.write_samples(each(pairs), bool(args.tqdm), len(pairs))

    pairs = get(args.in_root, 'val')
    out_split_dir = os.path.join(args.out_root, 'val')
    with StreamingDatasetWriter(out_split_dir, fields, args.shard_size_limit) as out:
        out.write_samples(each(pairs), bool(args.tqdm), len(pairs))


if __name__ == '__main__':
    main(parse_args())
