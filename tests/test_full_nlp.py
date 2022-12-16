# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import copy
import os

import pytest
import torch
from torch.utils.data import DataLoader
from torchmetrics.classification import Accuracy

from composer.algorithms import GatedLinearUnits
from composer.metrics.nlp import LanguageCrossEntropy, MaskedAccuracy
from composer.models import HuggingFaceModel
from composer.trainer import Trainer
from composer.utils import dist, inference, reproducibility
from tests.common.datasets import RandomTextClassificationDataset, RandomTextLMDataset
from tests.common.models import SimpleTransformerClassifier, SimpleTransformerMaskedLM


def pretraining_test_helper(tokenizer, model, algorithms, tmp_path):
    transformers = pytest.importorskip('transformers')

    pretraining_model_copy = copy.deepcopy(model)
    pretraining_train_dataset = RandomTextLMDataset(size=10,
                                                    vocab_size=tokenizer.vocab_size,
                                                    sequence_length=4,
                                                    use_keys=True)

    collator = transformers.DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)
    pretraining_train_dataloader = DataLoader(pretraining_train_dataset,
                                              batch_size=2,
                                              sampler=dist.get_sampler(pretraining_train_dataset),
                                              collate_fn=collator)
    pretraining_eval_dataloader = DataLoader(pretraining_train_dataset,
                                             batch_size=2,
                                             sampler=dist.get_sampler(pretraining_train_dataset),
                                             collate_fn=collator)

    pretraining_trainer = Trainer(model=pretraining_model_copy,
                                  train_dataloader=pretraining_train_dataloader,
                                  save_folder=str(tmp_path / 'pretraining_checkpoints'),
                                  max_duration='1ep',
                                  seed=17,
                                  algorithms=algorithms)
    pretraining_trainer.fit()
    reproducibility.seed_all(17)  # seed so that the masking is the same
    pretraining_trainer.eval(pretraining_eval_dataloader)

    loaded_pretraining_trainer = Trainer(model=model,
                                         load_path=str(tmp_path / 'pretraining_checkpoints' / 'latest-rank0.pt'),
                                         seed=17,
                                         algorithms=algorithms)

    reproducibility.seed_all(17)  # seed so that the masking is the same
    loaded_pretraining_trainer.eval(pretraining_eval_dataloader)

    original_ce = pretraining_trainer.state.eval_metrics['eval']['LanguageCrossEntropy']
    loaded_ce = loaded_pretraining_trainer.state.eval_metrics['eval']['LanguageCrossEntropy']
    assert original_ce.compute() > 0.0
    assert original_ce.compute() == loaded_ce.compute()

    return str(tmp_path / 'pretraining_checkpoints' / 'latest-rank0.pt')


def finetuning_test_helper(tokenizer, model, algorithms, checkpoint_path, tmp_path):
    finetuning_model_copy = copy.deepcopy(model)

    finetuning_train_dataset = RandomTextClassificationDataset(size=100,
                                                               vocab_size=tokenizer.vocab_size,
                                                               sequence_length=4,
                                                               num_classes=3,
                                                               use_keys=isinstance(model, HuggingFaceModel))
    finetuning_train_dataloader = DataLoader(finetuning_train_dataset,
                                             batch_size=10,
                                             sampler=dist.get_sampler(finetuning_train_dataset))
    finetuning_eval_dataloader = DataLoader(finetuning_train_dataset,
                                            batch_size=10,
                                            sampler=dist.get_sampler(finetuning_train_dataset))

    finetuning_trainer = Trainer(model=model,
                                 train_dataloader=finetuning_train_dataloader,
                                 save_folder=str(tmp_path / 'finetuning_checkpoints'),
                                 load_path=checkpoint_path,
                                 load_weights_only=True,
                                 max_duration='2ep',
                                 seed=17,
                                 algorithms=algorithms)
    finetuning_trainer.fit()
    finetuning_trainer.eval(finetuning_eval_dataloader)

    loaded_finetuning_trainer = Trainer(model=finetuning_model_copy,
                                        load_path=str(tmp_path / 'finetuning_checkpoints' / 'latest-rank0.pt'),
                                        seed=17,
                                        algorithms=algorithms)

    loaded_finetuning_trainer.eval(finetuning_eval_dataloader)

    original_acc = finetuning_trainer.state.eval_metrics['eval']['Accuracy']
    loaded_acc = loaded_finetuning_trainer.state.eval_metrics['eval']['Accuracy']
    assert original_acc.compute() > 0.0
    assert original_acc.compute() == loaded_acc.compute()

    return loaded_finetuning_trainer, finetuning_eval_dataloader


def inference_test_helper(model, original_input, original_output, tmp_path, save_format):
    os.mkdir(tmp_path / 'inference_checkpoints')
    sample_input = (original_input, {})

    inference.export_for_inference(model=model,
                                   save_format=save_format,
                                   save_path=str(tmp_path / 'inference_checkpoints' / f'exported_model.{save_format}'),
                                   sample_input=sample_input)

    copied_batch = copy.deepcopy(original_input)

    if save_format == 'onnx':
        onnx = pytest.importorskip('onnx')
        ort = pytest.importorskip('onnxruntime')
        loaded_inference_model = onnx.load(str(tmp_path / 'inference_checkpoints' / 'exported_model.onnx'))
        onnx.checker.check_model(loaded_inference_model)
        ort_session = ort.InferenceSession(str(tmp_path / 'inference_checkpoints' / 'exported_model.onnx'))

        for key, value in copied_batch.items():
            copied_batch[key] = value.numpy()
        loaded_model_out = ort_session.run(None, copied_batch)
    elif save_format == 'torchscript':
        loaded_inference_model = torch.jit.load(str(tmp_path / 'inference_checkpoints' / 'exported_model.torchscript'))
        loaded_inference_model.eval()
        loaded_model_out = loaded_inference_model(copied_batch)
    else:
        raise ValueError('Unsupported save format')

    torch.testing.assert_close(
        loaded_model_out[1] if isinstance(loaded_model_out, list) else loaded_model_out.detach().numpy(),
        original_output.detach().numpy()
        if isinstance(original_output, torch.Tensor) else original_output.logits.detach().numpy())


@pytest.mark.parametrize('model_type,algorithms,save_format', [('tinybert', [GatedLinearUnits()], 'onnx'),
                                                               ('simpletransformer', [], 'torchscript')])
def test_full_nlp_pipeline(model_type, algorithms, save_format, tiny_bert_tokenizer, tmp_path, request):

    tiny_bert_model = None
    if model_type == 'tinybert':
        tiny_bert_model = request.getfixturevalue('tiny_bert_model')

    # pretraining
    if model_type == 'tinybert':
        assert tiny_bert_model is not None
        pretraining_metrics = [
            LanguageCrossEntropy(ignore_index=-100, vocab_size=tiny_bert_tokenizer.vocab_size),
            MaskedAccuracy(ignore_index=-100)
        ]
        pretraining_model = HuggingFaceModel(tiny_bert_model,
                                             tiny_bert_tokenizer,
                                             use_logits=True,
                                             metrics=pretraining_metrics)
    elif model_type == 'simpletransformer':
        pretraining_model = SimpleTransformerMaskedLM(vocab_size=tiny_bert_tokenizer.vocab_size)
    else:
        raise ValueError('Unsupported model type')
    pretraining_output_path = pretraining_test_helper(tiny_bert_tokenizer, pretraining_model, algorithms, tmp_path)

    # finetuning
    if model_type == 'tinybert':
        finetuning_metric = Accuracy()
        hf_finetuning_model, _ = HuggingFaceModel.hf_from_composer_checkpoint(
            pretraining_output_path,
            model_instantiation_class='transformers.AutoModelForSequenceClassification',
            model_config_kwargs={'num_labels': 3})
        finetuning_model = HuggingFaceModel(model=hf_finetuning_model,
                                            tokenizer=tiny_bert_tokenizer,
                                            use_logits=True,
                                            metrics=[finetuning_metric])
    elif model_type == 'simpletransformer':
        finetuning_model = SimpleTransformerClassifier(vocab_size=tiny_bert_tokenizer.vocab_size, num_classes=3)
    else:
        raise ValueError('Unsupported model type.')
    finetuning_trainer, finetuning_dataloader = finetuning_test_helper(tiny_bert_tokenizer, finetuning_model,
                                                                       algorithms, pretraining_output_path, tmp_path)

    # inference
    batch = next(iter(finetuning_dataloader))
    finetuning_trainer.state.model.eval()
    original_output = finetuning_trainer.state.model(batch)
    inference_test_helper(finetuning_trainer.state.model, batch, original_output, tmp_path, save_format)
