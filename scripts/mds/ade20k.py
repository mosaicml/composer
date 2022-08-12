# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""ADE20K streaming dataset conversion scripts."""

import os
import random
from argparse import ArgumentParser, Namespace
from glob import glob
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


def get(in_root: str, split: str, shuffle: bool) -> List[Tuple[str, str, str]]:
    """Collect the samples for this dataset split.

    Args:
        in_root (str): Input dataset directory.
        split (str): Split name.
        shuffle (bool): Whether to shuffle the samples before writing.

    Returns:
        List of samples of (uid, image_filename, annotation_filename).
    """
    # Get uids
    image_glob_pattern = f'{in_root}/images/{split}/ADE_{split}_*.jpg'
    images = sorted(glob(image_glob_pattern))
    uids = [s.strip('.jpg')[-8:] for s in images]

    # Remove some known corrupted uids from 'train' split
    if split == 'train':
        corrupted_uids = ['00003020', '00001701', '00013508', '00008455']
        uids = [uid for uid in uids if uid not in corrupted_uids]

    # Create samples
    samples = [(uid, f'{in_root}/images/{split}/ADE_{split}_{uid}.jpg',
                f'{in_root}/annotations/{split}/ADE_{split}_{uid}.png') for uid in uids]

    # Optionally shuffle samples at dataset creation for extra randomness
    if shuffle:
        random.shuffle(samples)

    return samples


def each(samples: List[Tuple[str, str, str]]) -> Iterable[Dict[str, Any]]:
    """Generator over each dataset sample.

    Args:
        samples (list): List of samples of (uid, image_filename, annotation_filename).

    Yields:
        Sample dicts.
    """
    for (uid, image_file, annotation_file) in samples:
        uid = uid.encode('utf-8')
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
    fields = ['uid', 'image', 'annotation']

    for (split, expected_num_samples, shuffle) in [
        ('train', 20206, True),
        ('val', 2000, False),
    ]:
        # Get samples
        samples = get(in_root=args.in_root, split=split, shuffle=shuffle)
        assert len(samples) == expected_num_samples

        # Write samples
        with StreamingDatasetWriter(dirname=os.path.join(args.out_root, split),
                                    fields=fields,
                                    shard_size_limit=args.shard_size_limit) as out:
            out.write_samples(samples=each(samples), use_tqdm=bool(args.tqdm), total=len(samples))


if __name__ == '__main__':
    main(parse_args())
