# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""C4 streaming dataset conversion scripts."""

import os
import random
from argparse import ArgumentParser, Namespace
from glob import glob
from itertools import islice
from typing import Any, Dict, Iterable, List, Tuple

import datasets
import torch
from datasets import Dataset

from composer.datasets.streaming import StreamingDatasetWriter


def parse_args() -> Namespace:
    """Parse commandline arguments."""
    args = ArgumentParser()
    args.add_argument('--out_root', type=str, required=True)
    args.add_argument('--shard_size_limit', type=int, default=1 << 27)
    args.add_argument('--tqdm', type=int, default=1)
    return args.parse_args()


def get(split: str) -> Dataset:
    """Collect the samples for this dataset split.

    Args:
        split (str): Split name.

    Returns:
        A HF Dataset.
    """
    return datasets.load_dataset(path="c4", name="en", split=split)


def each(dataset: Dataset) -> Iterable[Dict[str, bytes]]:
    """Generator over each dataset sample.

    Args:
        samples (Dataset): A HF Dataset locally downloaded.

    Yields:
        Sample dicts.
    """
    num_workers = 64
    batch_size = 512
    prefetch_factor = max(1, 2 * batch_size // num_workers)

    loader = torch.utils.data.DataLoader(dataset=dataset,
                                         batch_size=batch_size,
                                         num_workers=num_workers,
                                         prefetch_factor=prefetch_factor)
    for batch in loader:
        keys = list(batch.keys())
        current_bs = len(batch[keys[0]])
        for idx in range(current_bs):
            yield {key: batch_values[idx].encode("utf-8") for key, batch_values in batch.items()}


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
        # Get dataset
        dataset = get(split=split)

        # Write samples
        with StreamingDatasetWriter(dirname=os.path.join(args.out_root, split_new_name),
                                    fields=fields,
                                    shard_size_limit=args.shard_size_limit) as out:
            out.write_samples(samples=each(dataset), use_tqdm=bool(args.tqdm), total=expected_num_samples)


if __name__ == '__main__':
    main(parse_args())
