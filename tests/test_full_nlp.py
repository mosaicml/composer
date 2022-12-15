# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import copy
import os

import pytest
import torch
from torch.utils.data import DataLoader
from torchmetrics.classification import Accuracy

from composer.metrics.nlp import LanguageCrossEntropy, MaskedAccuracy
from composer.models import HuggingFaceModel
from composer.trainer import Trainer
from composer.utils import dist, inference, reproducibility
from tests.common.datasets import RandomTextClassificationDataset, RandomTextLMDataset


# @pytest.mark.parameterize('model_type', ['tinybert', 'simpletransformer'])
def test_full_nlp_pipeline(tiny_bert_model, tiny_bert_tokenizer, tmp_path):
    transformers = pytest.importorskip('transformers')
    onnx = pytest.importorskip('onnx')
    ort = pytest.importorskip('onnxruntime')

    pretraining_metrics = [
        LanguageCrossEntropy(ignore_index=-100, vocab_size=tiny_bert_tokenizer.vocab_size),
        MaskedAccuracy(ignore_index=-100)
    ]
    pretraining_model = HuggingFaceModel(tiny_bert_model,
                                         tiny_bert_tokenizer,
                                         use_logits=True,
                                         metrics=pretraining_metrics)
    pretraining_train_dataset = RandomTextLMDataset(size=10,
                                                    vocab_size=tiny_bert_tokenizer.vocab_size,
                                                    sequence_length=4,
                                                    use_keys=True)

    collator = transformers.DataCollatorForLanguageModeling(tokenizer=tiny_bert_tokenizer, mlm_probability=0.15)
    pretraining_train_dataloader = DataLoader(pretraining_train_dataset,
                                              batch_size=2,
                                              sampler=dist.get_sampler(pretraining_train_dataset),
                                              collate_fn=collator)
    pretraining_eval_dataloader = DataLoader(pretraining_train_dataset,
                                             batch_size=2,
                                             sampler=dist.get_sampler(pretraining_train_dataset),
                                             collate_fn=collator)

    pretraining_trainer = Trainer(model=pretraining_model,
                                  train_dataloader=pretraining_train_dataloader,
                                  save_folder=str(tmp_path / 'pretraining_checkpoints'),
                                  max_duration='1ep',
                                  seed=17)
    pretraining_trainer.fit()
    reproducibility.seed_all(17)  # seed so that the masking is the same
    pretraining_trainer.eval(pretraining_eval_dataloader)

    loaded_pretraining_trainer = Trainer(model=pretraining_model,
                                         load_path=str(tmp_path / 'pretraining_checkpoints' / 'latest-rank0.pt'),
                                         seed=17)
    reproducibility.seed_all(17)  # seed so that the masking is the same
    loaded_pretraining_trainer.eval(pretraining_eval_dataloader)

    original_ce = pretraining_trainer.state.eval_metrics['eval']['LanguageCrossEntropy']
    loaded_ce = loaded_pretraining_trainer.state.eval_metrics['eval']['LanguageCrossEntropy']
    assert original_ce.compute() > 0.0
    assert original_ce.compute() == loaded_ce.compute()

    finetuning_metrics = Accuracy()
    reproducibility.seed_all(17)
    hf_finetuning_model, finetuning_tokenizer = HuggingFaceModel.hf_from_composer_checkpoint(
        str(tmp_path / 'pretraining_checkpoints' / 'latest-rank0.pt'),
        model_instantiation_class='transformers.AutoModelForSequenceClassification',
        model_config_kwargs={'num_labels': 3})
    finetuning_model = HuggingFaceModel(model=hf_finetuning_model,
                                        tokenizer=finetuning_tokenizer,
                                        use_logits=True,
                                        metrics=[finetuning_metrics])

    finetuning_train_dataset = RandomTextClassificationDataset(size=100,
                                                               vocab_size=tiny_bert_tokenizer.vocab_size,
                                                               sequence_length=4,
                                                               num_classes=3,
                                                               use_keys=True)
    finetuning_train_dataloader = DataLoader(finetuning_train_dataset,
                                             batch_size=10,
                                             sampler=dist.get_sampler(finetuning_train_dataset))
    finetuning_eval_dataloader = DataLoader(finetuning_train_dataset,
                                            batch_size=10,
                                            sampler=dist.get_sampler(finetuning_train_dataset))
    finetuning_trainer = Trainer(model=finetuning_model,
                                 train_dataloader=finetuning_train_dataloader,
                                 save_folder=str(tmp_path / 'finetuning_checkpoints'),
                                 max_duration='2ep',
                                 seed=17)
    finetuning_trainer.fit()
    finetuning_trainer.eval(finetuning_eval_dataloader)

    loaded_finetuning_trainer = Trainer(model=finetuning_model,
                                        load_path=str(tmp_path / 'finetuning_checkpoints' / 'latest-rank0.pt'),
                                        seed=17)
    loaded_finetuning_trainer.eval(finetuning_eval_dataloader)

    original_acc = finetuning_trainer.state.eval_metrics['eval']['Accuracy']
    loaded_acc = loaded_finetuning_trainer.state.eval_metrics['eval']['Accuracy']
    assert original_acc.compute() > 0.0
    assert original_acc.compute() == loaded_acc.compute()

    os.mkdir(tmp_path / 'inference_checkpoints')
    inference.export_for_inference(model=loaded_finetuning_trainer.state.model,
                                   save_format='onnx',
                                   save_path=str(tmp_path / 'inference_checkpoints' / 'exported_model.onnx'),
                                   sample_input=(next(iter(finetuning_train_dataloader)), {}))

    loaded_inference_model = onnx.load(str(tmp_path / 'inference_checkpoints' / 'exported_model.onnx'))
    onnx.checker.check_model(loaded_inference_model)
    ort_session = ort.InferenceSession(str(tmp_path / 'inference_checkpoints' / 'exported_model.onnx'))

    batch = next(iter(finetuning_eval_dataloader))
    copied_batch = copy.deepcopy(batch)

    loaded_finetuning_trainer.state.model.eval()
    out_original = loaded_finetuning_trainer.state.model(batch)

    for key, value in copied_batch.items():
        copied_batch[key] = value.numpy()
    loaded_model_out = ort_session.run(None, copied_batch)

    assert out_original.loss.detach().numpy() == loaded_model_out[0]
    torch.testing.assert_close(loaded_model_out[1], out_original.logits.detach().numpy())

    # add algo (GLU)
    # parametrize
    assert False
