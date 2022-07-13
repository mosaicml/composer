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
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from composer.datasets.streaming import StreamingDatasetWriter


def parse_args() -> Namespace:
    """Parse commandline arguments."""
    args = ArgumentParser()
    args.add_argument('--out_root', type=str, required=True)
    args.add_argument('--shard_size_limit', type=int, default=1 << 28)
    args.add_argument('--tqdm', type=int, default=1)
    return args.parse_args()


def get(split: str) -> IterableDataset:
    """Collect the samples for this dataset split.

    Args:
        split (str): Split name.

    Returns:
        An IterableDataset.
    """

    class ShardedC4(IterableDataset):

        def __init__(self):
            self.dataset = datasets.load_dataset(path='c4', name='en', split=split, streaming=True)

        def num_shards(self):
            return len(self.dataset._ex_iterable.kwargs['filepaths'])

        def __iter__(self):
            worker_info = get_worker_info()
            if worker_info:
                num_workers = worker_info.num_workers
                worker_id = worker_info.id
                shards = self.dataset._ex_iterable.kwargs['filepaths']
                assert len(shards) % num_workers == 0
                self.dataset._ex_iterable.kwargs['filepaths'] = shards[worker_id::num_workers]
            return iter(self.dataset)

    return ShardedC4()


def each(dataset: IterableDataset) -> Iterable[Dict[str, bytes]]:
    """Generator over each dataset sample.

    Args:
        samples (Dataset): A HF Dataset locally downloaded.

    Yields:
        Sample dicts.
    """
    num_workers = min(64, dataset.num_shards())
    batch_size = 512
    # If using multiple workers, configure each worker to prefetch as many samples as it can, up to the aggregate device batch size
    # If not using workers, the torch DataLoader expects the default value for prefetch_factor, which non-intuitively must be 2.
    prefetch_factor = max(1, 2 * batch_size // num_workers) if num_workers > 0 else 2

    loader = DataLoader(
        dataset=dataset,
        sampler=None,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
    )
    for batch in loader:
        keys = list(batch.keys())
        current_bs = len(batch[keys[0]])
        for idx in range(current_bs):
            yield {key: batch_values[idx].encode('utf-8') for key, batch_values in batch.items()}


def main(args: Namespace) -> None:
    """Main: create C4 streaming dataset.

    Args:
        args (Namespace): Commandline arguments.
    """
    fields = ['text', 'timestamp', 'url']

    for (split, split_new_name, expected_num_samples) in [
        ('train', 'train', 364868892),
        ('validation', 'val', 364608),
    ]:
        # Get dataset
        dataset = get(split=split)

        # Write samples
        with StreamingDatasetWriter(dirname=os.path.join(args.out_root, split_new_name),
                                    fields=fields,
                                    shard_size_limit=args.shard_size_limit,
                                    compression=None) as out:
            out.write_samples(samples=each(dataset), use_tqdm=bool(args.tqdm), total=expected_num_samples)


if __name__ == '__main__':
    main(parse_args())
