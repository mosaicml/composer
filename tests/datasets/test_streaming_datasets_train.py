# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from composer.metrics.nlp import LanguageCrossEntropy, MaskedAccuracy
from composer.models import HuggingFaceModel
from composer.trainer import Trainer
from tests.common import device, world_size


@device('cpu', 'gpu')
@world_size(1, 2)
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
def test_streaming_datasets(dataset, dataset_args, seed, tiny_bert_tokenizer, tiny_bert_model, tmp_path, world_size,
                            device):
    if torch.cuda.is_available() and device == 'cpu' or world_size > 1 and device == 'cpu':
        pytest.xfail('There is currently a bug in streaming that prevents this combination of settings from working.')

    streaming = pytest.importorskip('streaming')
    transformers = pytest.importorskip('transformers')
    name_to_cls = {
        'c4': streaming.text.c4.StreamingC4,
        'pile': streaming.text.pile.StreamingPile,
        'enwiki': streaming.text.enwiki.StreamingEnWiki
    }

    # This seed setting is necessary to prevent a shared memory collision due to a streaming bug
    np.random.seed(seed)
    streaming_dataset = name_to_cls[dataset](local=str(tmp_path / dataset),
                                             split='val',
                                             predownload=None,
                                             batch_size=8,
                                             **dataset_args)

    pretraining_metrics = [
        LanguageCrossEntropy(ignore_index=-100, vocab_size=tiny_bert_tokenizer.vocab_size),
        MaskedAccuracy(ignore_index=-100)
    ]
    model = HuggingFaceModel(model=tiny_bert_model, use_logits=True, metrics=pretraining_metrics)
    collator = transformers.DataCollatorForLanguageModeling(tokenizer=tiny_bert_tokenizer,
                                                            mlm_probability=0.15) if dataset != 'enwiki' else None
    dataloader = DataLoader(streaming_dataset, batch_size=8, collate_fn=collator)

    trainer = Trainer(model=model, train_dataloader=dataloader, max_duration='2ba', device=device)
    trainer.fit()
