# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import transformers
from torch.utils.data import DataLoader

from composer.metrics.nlp import LanguageCrossEntropy, MaskedAccuracy
from composer.models import HuggingFaceModel
from composer.trainer import Trainer
from composer.utils import dist, reproducibility
from tests.common.datasets import RandomTextLMDataset


# @pytest.mark.parameterize('model_type', ['tinybert', 'simpletransformer'])
def test_full_nlp_pipeline(tiny_bert_model, tiny_bert_tokenizer, tmp_path):

    metrics = [
        LanguageCrossEntropy(ignore_index=-100, vocab_size=tiny_bert_tokenizer.vocab_size),
        MaskedAccuracy(ignore_index=-100)
    ]
    model = HuggingFaceModel(tiny_bert_model, tiny_bert_tokenizer, use_logits=True, metrics=metrics)
    pretraining_train_dataset = RandomTextLMDataset(size=100,
                                                    vocab_size=tiny_bert_tokenizer.vocab_size,
                                                    sequence_length=4,
                                                    use_keys=True)

    collator = transformers.DataCollatorForLanguageModeling(tokenizer=tiny_bert_tokenizer, mlm_probability=0.15)
    pretraining_train_dataloader = DataLoader(pretraining_train_dataset,
                                              batch_size=10,
                                              sampler=dist.get_sampler(pretraining_train_dataset),
                                              collate_fn=collator)
    pretraining_eval_dataloader = DataLoader(pretraining_train_dataset,
                                             batch_size=10,
                                             sampler=dist.get_sampler(pretraining_train_dataset),
                                             collate_fn=collator)

    pretraining_trainer = Trainer(
        model=model,
        train_dataloader=pretraining_train_dataloader,
        save_folder=str(tmp_path / 'pretraining_checkpoints'),
        max_duration='2ep',
    )
    pretraining_trainer.fit()
    reproducibility.seed_all(17)
    pretraining_trainer.eval(pretraining_eval_dataloader)

    loaded_pretraining_trainer = Trainer(model=model,
                                         load_path=str(tmp_path / 'pretraining_checkpoints' / 'latest-rank0.pt'))
    reproducibility.seed_all(17)
    loaded_pretraining_trainer.eval(pretraining_eval_dataloader)

    original_masked_acc = pretraining_trainer.state.eval_metrics['eval']['MaskedAccuracy']
    loaded_masked_acc = loaded_pretraining_trainer.state.eval_metrics['eval']['MaskedAccuracy']
    assert original_masked_acc.compute() == loaded_masked_acc.compute()
    # pretrain base model, checkpoint, eval
    # load random classification dataset
    # load for finetuning, checkpoint, eval
    # load for inference, eval
    # add algo
    assert False
