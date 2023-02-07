# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import pytest
from torch.utils.data import DataLoader

from composer.models.gpt2 import create_gpt2
from composer.trainer import Trainer
from tests.common.datasets import RandomTextLMDataset


def test_gpt2_hf_factory(tiny_gpt2_config, tiny_gpt2_tokenizer, monkeypatch):
    transformers = pytest.importorskip('transformers')
    monkeypatch.setattr('transformers.AutoConfig.from_pretrained', lambda x: tiny_gpt2_config)
    gpt2_composer_model = create_gpt2(use_pretrained=False,
                                      pretrained_model_name='dummy',
                                      model_config=None,
                                      tokenizer_name=None,
                                      gradient_checkpointing=False)

    train_dataset = RandomTextLMDataset(size=8,
                                        vocab_size=tiny_gpt2_tokenizer.vocab_size,
                                        sequence_length=8,
                                        use_keys=True)
    collator = transformers.DataCollatorForLanguageModeling(tokenizer=tiny_gpt2_tokenizer, mlm=False)
    train_dataloader = DataLoader(train_dataset, batch_size=4, collate_fn=collator)

    trainer = Trainer(model=gpt2_composer_model, train_dataloader=train_dataloader, max_duration='1ep')
    trainer.fit()

    assert trainer.state.train_metrics['LanguagePerplexity'].compute() > 0.0
