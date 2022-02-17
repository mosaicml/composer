# Copyright 2021 MosaicML. All Rights Reserved.

from typing import Optional

import pytest
import torch

from composer.datasets.dataloader import DataloaderHparams
from composer.datasets.streaming_lm_datasets import StreamingLMDatasetHparams

dataset_hparams = StreamingLMDatasetHparams(
    dataset_name="c4",
    dataset_config_name="en",
    split="validation",
    max_shards=-1,
    max_samples=100,
    tokenizer_name="bert-base-uncased",
    max_seq_len=10,
    group_method="truncate",
    use_masked_lm=True,
    mlm_probability=0.15,
    seed=10,
    shuffle=False,
    drop_last=True,
)

print("Approx Samples", dataset_hparams._get_approx_num_samples())
print("Approx Tokens", dataset_hparams._get_approx_num_tokens())

dataloader_hparams = DataloaderHparams(
    num_workers=1,
    prefetch_factor=1,
    persistent_workers=1,
    pin_memory=True,
    timeout=0,
)
batch_size = 8

dataloader_spec = dataset_hparams.initialize_object(
    batch_size=batch_size,
    dataloader_hparams=dataloader_hparams,
)

dataloader, _, _ = dataloader_spec

samples = 0
for i, batch in enumerate(dataloader):
    if i == 0:
        print(batch)
    samples += batch['input_ids'].shape[0]
print(samples)
