# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import os
import random
from argparse import ArgumentParser, Namespace
from glob import glob
from itertools import islice
from typing import Any, Dict, Iterable, List, Tuple

import datasets
from datasets import IterableDataset

from composer.datasets.streaming import StreamingDatasetWriter


def parse_args() -> Namespace:
    """Parse commandline arguments."""
    args = ArgumentParser()
    args.add_argument('--out_root', type=str, required=True)
    args.add_argument('--shard_size_limit', type=int, default=1 << 27)
    args.add_argument('--tqdm', type=int, default=1)
    return args.parse_args()

def get(split: str) -> IterableDataset:
    """Collect the samples for this dataset split.

    Args:
        split (str): Split name.

    Returns:
        A HF IterableDataset.
    """
    return datasets.load_dataset(path="c4", name="en", split=split, streaming=True)


def each(samples: IterableDataset) -> Iterable[Dict[str, bytes]]:
    """Generator over each dataset sample.

    Args:
        samples (list): List of samples of (uid, image_filename, annotation_filename).

    Yields:
        Sample dicts.
    """
    for sample in samples:
        yield {
            key: value.encode("utf-8")
            for key, value in sample.items()
        }


def main(args: Namespace) -> None:
    """Main: create C4 streaming dataset.

    Args:
        args (Namespace): Commandline arguments.
    """
    fields = ["text", "timestamp", "url"]

    for (split, split_new_name, expected_num_samples) in [
        ("train", "train", 364868892),
        ("validation", "val", 364608),
    ]:
        # Get samples
        samples = get(split=split)

        # Write samples
        with StreamingDatasetWriter(dirname=os.path.join(args.out_root, split_new_name),
                                    fields=fields,
                                    shard_size_limit=args.shard_size_limit) as out:
            out.write_samples(samples=each(samples), use_tqdm=bool(args.tqdm), total=expected_num_samples)


if __name__ == '__main__':
    main(parse_args())
