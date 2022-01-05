# Copyright 2021 MosaicML. All Rights Reserved.

from typing import Optional

import pytest

from composer.datasets.dataloader import DataloaderHparams
from composer.datasets.streaming_lm_datasets import StreamingLMDatasetHparams


@pytest.mark.parametrize('split', [
    "train",
    "validation",
])
def test_streaming_lm(split):

    dataset_hparams = StreamingLMDatasetHparams(
        dataset_name="c4",
        dataset_config_name="en",
        split=split,
        tokenizer_name="gpt2",
        max_seq_len="2048",
        collate_method="concat",
        use_masked_lm=False,
        mlm_probability=0.15,
        seed=10,
        shuffle=False,
        drop_last=True,
    )

    # dataloader_hparams = DataloaderHparams(
    #     num_workers=1,
    #     prefetch_factor=1,
    #     persistent_workers=1,
    #     pin_memory=True,
    #     timeout=0,
    # )

    # batch_size = 8
    # dataloader_spec = dataset_hparams.initialize_object(
    #     batch_size=batch_size,
    #     dataloader_hparams=dataloader_hparams,
    # )

    print("Approx Samples", dataset_hparams.get_approx_num_samples())
    print("Approx Tokens", dataset_hparams.get_approx_num_tokens())

