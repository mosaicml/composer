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
        List of samples of (uid, image filename, annotation filename).
    """

    # Get uids
    image_glob_pattern = f'{in_root}/images/{split}/ADE_{split}_*.jpg'
    images = sorted(glob(image_glob_pattern))
    uids = [s.strip(".jpg")[-8:] for s in images]

    # Remove some known corrupted uids from 'train' split
    if split == "train":
        corrupted_uids = ['00003020', '00001701', '00013508', '00008455']
        uids = [uid for uid in uids if uid not in corrupted_uids]

    # Create and shuffle samples
    samples = [(uid, f'{in_root}/images/{split}/ADE_{split}_{uid}.jpg',
                f'{in_root}/annotations/{split}/ADE_{split}_{uid}.png') for uid in uids]

    # Shuffle samples at dataset creation for extra randomness
    shuffle(samples)

    return samples


def each(samples: List[Tuple[str, int]]) -> Iterable[Dict[str, Any]]:
    """Generator over each dataset sample.

    Args:
        samples (list): List of samples of (uid, image filename, annotation filename).

    Yields:
        Sample dicts.
    """
    for (uid, image_file, annotation_file) in samples:
        uid = uid.encode("utf-8")
        image = open(image_file, 'rb').read()
        annotation = open(annotation_file, 'rb').read()
        yield {
            'uid': uid,
            'image': image,
            'annotation': annotation,
        }


def main(args: Namespace) -> None:
    """Main: create ADE20K streaming dataset.

    Args:
        args (Namespace): Commandline arguments.
    """
    fields = 'uid', 'image', 'annotation'

    # Get train samples
    train_samples = get(args.in_root, 'train')
    assert len(train_samples) == 20206

    # Write train samples
    out_split_dir = os.path.join(args.out_root, 'train')
    with StreamingDatasetWriter(out_split_dir, fields, args.shard_size_limit) as out:
        out.write_samples(each(train_samples), bool(args.tqdm), len(train_samples))

    # Get val samples
    val_samples = get(args.in_root, 'val')
    assert len(val_samples) == 2000

    # Write val samples
    out_split_dir = os.path.join(args.out_root, 'val')
    with StreamingDatasetWriter(out_split_dir, fields, args.shard_size_limit) as out:
        out.write_samples(each(val_samples), bool(args.tqdm), len(val_samples))


if __name__ == '__main__':
    main(parse_args())
