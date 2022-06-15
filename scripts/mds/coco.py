# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Coco streaming dataset conversion scripts."""

import os
from argparse import ArgumentParser, Namespace
from typing import Dict, Iterable

import numpy as np

from composer.datasets.coco import COCODetection
from composer.datasets.streaming import StreamingDatasetWriter


def parse_args() -> Namespace:
    """Parse commandline arguments.

    Returns:
        Namespace: Commandline arguments.
    """
    args = ArgumentParser()
    args.add_argument('--in_root', type=str, required=True)
    args.add_argument('--out_root', type=str, required=True)
    args.add_argument('--shard_size_limit', type=int, default=1 << 25)
    args.add_argument('--tqdm', type=int, default=1)
    return args.parse_args()


def each(dataset: COCODetection) -> Iterable[Dict[str, bytes]]:
    """Generator over each dataset sample.

    Args:
        dataset (COCODetection): COCO detection dataset.

    Yields:
        Sample dicts.
    """
    indices = np.random.permutation(len(dataset))
    for idx in indices:
        _, img_id, (htot, wtot), bbox_sizes, bbox_labels = dataset[idx]

        img_id = dataset.img_keys[idx]
        img_data = dataset.images[img_id]
        img_basename = img_data[0]
        img_filename = os.path.join(dataset.img_folder, img_basename)
        img_bytes = open(img_filename, 'rb').read()

        yield {
            'img': img_bytes,
            'img_id': np.int64(img_id).tobytes(),
            'htot': np.int64(htot).tobytes(),
            'wtot': np.int64(wtot).tobytes(),
            'bbox_sizes': bbox_sizes.numpy().tobytes(),  # (_, 4) float32.
            'bbox_labels': bbox_labels.numpy().tobytes(),  # int64.
        }


def main(args: Namespace) -> None:
    """Main: create COCO streaming dataset.

    Args:
        args (Namespace): Commandline arguments.
    """
    fields = ['img', 'img_id', 'htot', 'wtot', 'bbox_sizes', 'bbox_labels']
    for (split, expected_num_samples, shuffle) in [
        ('train', 117266, True),
        ('val', 4952, False),
    ]:
        split_dir = os.path.join(args.out_root, split)

        img_folder = os.path.join(args.in_root, f'{split}2017')
        annotate_file = os.path.join(args.in_root, f'annotations/instances_{split}2017.json')
        dataset = COCODetection(img_folder, annotate_file)

        with StreamingDatasetWriter(split_dir, fields, args.shard_size_limit) as out:
            out.write_samples(each(dataset), bool(args.tqdm), len(dataset))


if __name__ == '__main__':
    main(parse_args())
