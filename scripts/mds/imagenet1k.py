# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""ImageNet 1K Streaming Dataset Conversion Script."""

import os
import random
from argparse import ArgumentParser, Namespace
from glob import glob
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


def get(split_dir: str, shuffle: bool) -> List[Tuple[str, int]]:
    """Collect the samples for this dataset split.

    Args:
        split_dir (str): Input dataset split directory.
        shuffle (bool): Whether to shuffle the samples before writing.

    Returns:
        List of pairs of (image_filename, class_id).
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
    if shuffle:
        random.shuffle(pairs)
    return pairs


def each(pairs: List[Tuple[str, int]]) -> Iterable[Dict[str, Any]]:
    """Generator over each dataset sample.

    Args:
        pairs (list): List of pairs of (image_filename, class_id).

    Yields:
        Sample dicts.
    """
    for image_filename, class_id in pairs:
        uid = image_filename.strip('.JPEG')[-8:]
        assert len(uid) == 8
        image = open(image_filename, 'rb').read()
        yield {
            'uid': uid.encode('utf-8'),
            'x': image,
            'y': np.int64(class_id).tobytes(),
        }


def main(args: Namespace) -> None:
    """Create ImageNet1k streaming dataset.

    Args:
        args (Namespace): Commandline arguments.
    """
    fields = ['uid', 'x', 'y']

    for (split, expected_num_samples, shuffle) in [
        ('train', 1281167, True),
        ('val', 50000, False),
    ]:
        # Get samples
        split_dir = os.path.join(args.in_root, split)
        samples = get(split_dir=split_dir, shuffle=shuffle)
        assert len(samples) == expected_num_samples

        # Write samples
        with StreamingDatasetWriter(dirname=os.path.join(args.out_root, split),
                                    fields=fields,
                                    shard_size_limit=args.shard_size_limit) as out:
            out.write_samples(samples=each(samples), use_tqdm=bool(args.tqdm), total=len(samples))


if __name__ == '__main__':
    main(parse_args())
