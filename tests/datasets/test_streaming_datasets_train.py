# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import pytest
from torch.utils.data import DataLoader

from composer.metrics.nlp import LanguageCrossEntropy, MaskedAccuracy
from composer.models import HuggingFaceModel
from composer.trainer import Trainer
from tests.common import world_size


@world_size(1, 2)
@pytest.mark.parametrize('dataset,dataset_args', [('c4', {
    'remote': 's3://mosaicml-internal-dataset-c4/mds/2/',
    'tokenizer_name': 'prajjwal1/bert-tiny',
    'max_seq_len': 256,
    'group_method': 'truncate'
}),
                                                  ('pile', {
                                                      'remote': 's3://mosaicml-internal-dataset-the-pile/mds/2/',
                                                      'tokenizer_name': 'prajjwal1/bert-tiny',
                                                      'max_seq_len': 256,
                                                      'group_method': 'truncate'
                                                  }),
                                                  ('enwiki', {
                                                      'remote': 's3://mosaicml-internal-dataset-enwiki-20200101/mds/2/'
                                                  })])
def test_streaming_datasets(dataset, dataset_args, tiny_bert_tokenizer, tiny_bert_model, tmp_path, world_size):
    streaming = pytest.importorskip('streaming')
    transformers = pytest.importorskip('transformers')
    name_to_cls = {
        'c4': streaming.text.c4.StreamingC4,
        'pile': streaming.text.pile.StreamingPile,
        'enwiki': streaming.text.enwiki.StreamingEnWiki
    }
    streaming_dataset = name_to_cls[dataset](local=str(tmp_path / dataset),
                                             split='val',
                                             predownload=10,
                                             batch_size=8,
                                             **dataset_args)

    pretraining_metrics = [
        LanguageCrossEntropy(ignore_index=-100, vocab_size=tiny_bert_tokenizer.vocab_size),
        MaskedAccuracy(ignore_index=-100)
    ]
    model = HuggingFaceModel(model=tiny_bert_model, use_logits=True, metrics=pretraining_metrics)
    collator = transformers.DataCollatorForLanguageModeling(tokenizer=tiny_bert_tokenizer, mlm_probability=0.15)
    dataloader = DataLoader(streaming_dataset, batch_size=8, collate_fn=collator)

    trainer = Trainer(model=model, train_dataloader=dataloader, max_duration='2ba')
    trainer.fit()
