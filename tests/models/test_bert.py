# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import pytest
from torch.utils.data import DataLoader

from composer.models.bert import create_bert_classification, create_bert_mlm
from composer.trainer import Trainer
from tests.common.datasets import RandomTextClassificationDataset, RandomTextLMDataset


def test_bert_mlm_hf_factory(tiny_bert_config, tiny_bert_tokenizer, monkeypatch):
    transformers = pytest.importorskip('transformers')
    monkeypatch.setattr('transformers.AutoConfig.from_pretrained', lambda x: tiny_bert_config)
    bert_composer_model = create_bert_mlm(use_pretrained=False,
                                          pretrained_model_name='dummy',
                                          model_config=None,
                                          tokenizer_name=None,
                                          gradient_checkpointing=False)

    train_dataset = RandomTextLMDataset(size=8,
                                        vocab_size=tiny_bert_tokenizer.vocab_size,
                                        sequence_length=8,
                                        use_keys=True)
    collator = transformers.DataCollatorForLanguageModeling(tokenizer=tiny_bert_tokenizer,
                                                            mlm=True,
                                                            mlm_probability=0.15)
    train_dataloader = DataLoader(train_dataset, batch_size=4, collate_fn=collator)

    trainer = Trainer(model=bert_composer_model, train_dataloader=train_dataloader, max_duration='1ep')
    trainer.fit()

    assert trainer.state.train_metrics['LanguageCrossEntropy'].compute() > 0.0


def test_bert_classification_hf_factory(tiny_bert_config, tiny_bert_tokenizer, monkeypatch):
    pytest.importorskip('transformers')

    def config_patch(x, num_labels):
        tiny_bert_config.num_labels = num_labels
        return tiny_bert_config

    monkeypatch.setattr('transformers.AutoConfig.from_pretrained', config_patch)
    bert_composer_model = create_bert_classification(use_pretrained=False,
                                                     pretrained_model_name='dummy',
                                                     model_config=None,
                                                     tokenizer_name=None,
                                                     gradient_checkpointing=False,
                                                     num_labels=3)

    train_dataset = RandomTextClassificationDataset(size=8,
                                                    vocab_size=tiny_bert_tokenizer.vocab_size,
                                                    sequence_length=8,
                                                    num_classes=3,
                                                    use_keys=True)
    train_dataloader = DataLoader(train_dataset, batch_size=4)

    trainer = Trainer(model=bert_composer_model, train_dataloader=train_dataloader, max_duration='1ep')
    trainer.fit()

    assert trainer.state.train_metrics['MulticlassAccuracy'].compute() > 0.0
