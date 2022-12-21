# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from composer.metrics.nlp import LanguageCrossEntropy, MaskedAccuracy
from composer.models import HuggingFaceModel
from composer.trainer import Trainer
from composer.utils import dist
from tests.common import device, world_size


@device('cpu', 'gpu')
@world_size(1, 2)
@pytest.mark.parametrize('num_workers', [0, 1, 2])
@pytest.mark.parametrize('dataset,dataset_args,seed',
                         [('c4', {
                             'remote': 's3://mosaicml-internal-dataset-c4/mds/2/',
                             'tokenizer_name': 'bert-base-uncased',
                             'max_seq_len': 256,
                             'group_method': 'truncate'
                         }, 1),
                          ('pile', {
                              'remote': 's3://mosaicml-internal-dataset-the-pile/mds/2/',
                              'tokenizer_name': 'bert-base-uncased',
                              'max_seq_len': 256,
                              'group_method': 'truncate'
                          }, 2), ('enwiki', {
                              'remote': 's3://mosaicml-internal-dataset-enwiki-20200101/mds/2b/'
                          }, 3)])
def test_streaming_datasets(num_workers, dataset, dataset_args, seed, tiny_bert_tokenizer, tiny_bert_model, world_size,
                            device, tmp_path):
    # Need to initialize dist before we get to streaming, because streaming always uses NCCL
    if not dist.is_initialized() and world_size > 1:
        dist.initialize_dist(device=device)

    from sys import platform

    if num_workers == 2 and device == 'gpu' and world_size == 1:
        pytest.xfail("don't know. fatal python error")

    if num_workers > 0 and device == 'gpu' and world_size == 2:
        pytest.xfail("don't know. various inconsistent hangs and failures")

    streaming = pytest.importorskip('streaming')
    transformers = pytest.importorskip('transformers')
    name_to_cls = {
        'c4': streaming.text.c4.StreamingC4,
        'pile': streaming.text.pile.StreamingPile,
        'enwiki': streaming.text.enwiki.StreamingEnWiki
    }

    # distribute the local dataset path from rank 0
    local_path = [os.path.abspath(tmp_path)]
    dist.broadcast_object_list(local_path, src=0)
    local_path = Path(local_path[0]) / dataset

    # This seed setting is necessary to prevent a shared memory collision due to a streaming bug
    np.random.seed(seed + (num_workers + 1) * 10 + (world_size + 1) * 100 + (1 if device == 'cpu' else 2) * 1000)
    streaming_dataset = name_to_cls[dataset](local=local_path, split='val', predownload=1, batch_size=8, **dataset_args)

    pretraining_metrics = [
        LanguageCrossEntropy(ignore_index=-100, vocab_size=tiny_bert_tokenizer.vocab_size),
        MaskedAccuracy(ignore_index=-100)
    ]
    model = HuggingFaceModel(model=tiny_bert_model, use_logits=True, metrics=pretraining_metrics)
    collator = transformers.DataCollatorForLanguageModeling(tokenizer=tiny_bert_tokenizer,
                                                            mlm_probability=0.15) if dataset != 'enwiki' else None
    dataloader = DataLoader(streaming_dataset, batch_size=8, num_workers=num_workers, collate_fn=collator)

    trainer = Trainer(model=model, train_dataloader=dataloader, max_duration='2ba', device=device)

    try:
        backend = torch.distributed.get_backend()  # type: ignore
    except:
        backend = None

    if platform == 'darwin' and num_workers > 0:
        pytest.xfail('Streaming currently does not work on OSX with subprocess workers.')

    trainer.fit()
