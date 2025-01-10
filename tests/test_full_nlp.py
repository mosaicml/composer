# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import copy
import os

import pytest
import torch
from packaging import version
from torch.utils.data import DataLoader
from torchmetrics.classification import MulticlassAccuracy
from transformers import BertConfig, BertForMaskedLM, BertForSequenceClassification, BertTokenizerFast

from composer.algorithms import GatedLinearUnits
from composer.loggers import RemoteUploaderDownloader
from composer.metrics.nlp import LanguageCrossEntropy, MaskedAccuracy
from composer.models import HuggingFaceModel
from composer.trainer import Trainer
from composer.utils import dist, get_device, inference, reproducibility
from tests.common import device
from tests.common.datasets import RandomTextClassificationDataset, RandomTextLMDataset
from tests.common.models import SimpleTransformerClassifier, SimpleTransformerMaskedLM


def get_model_embeddings(model):
    if isinstance(model, HuggingFaceModel):
        return model.model.bert.embeddings.word_embeddings.weight
    elif isinstance(model, SimpleTransformerClassifier) or isinstance(model, SimpleTransformerMaskedLM):
        return model.transformer_base.embedding.weight
    else:
        raise ValueError('Unsure how to get embeddings layer from model.')


def pretraining_test_helper(tokenizer, model, algorithms, tmp_path, device):
    transformers = pytest.importorskip('transformers')

    pretraining_model_copy = copy.deepcopy(model)
    pretraining_train_dataset = RandomTextLMDataset(
        size=4,
        vocab_size=tokenizer.vocab_size,
        sequence_length=2,
        use_keys=True,
    )

    collator = transformers.DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)
    pretraining_train_dataloader = DataLoader(
        pretraining_train_dataset,
        batch_size=2,
        sampler=dist.get_sampler(pretraining_train_dataset),
        collate_fn=collator,
    )
    pretraining_eval_dataloader = DataLoader(
        pretraining_train_dataset,
        batch_size=2,
        sampler=dist.get_sampler(pretraining_train_dataset),
        collate_fn=collator,
    )

    pretraining_trainer = Trainer(
        model=pretraining_model_copy,
        train_dataloader=pretraining_train_dataloader,
        save_folder=str(tmp_path / 'pretraining_checkpoints'),
        max_duration='2ba',
        seed=17,
        algorithms=algorithms,
        device=device,
    )
    pretraining_trainer.fit()
    reproducibility.seed_all(17)  # seed so that the masking is the same
    pretraining_trainer.eval(pretraining_eval_dataloader)

    loaded_pretraining_trainer = Trainer(
        model=model,
        load_path=str(tmp_path / 'pretraining_checkpoints' / 'latest-rank0.pt'),
        seed=17,
        algorithms=algorithms,
        device=device,
    )

    reproducibility.seed_all(17)  # seed so that the masking is the same
    loaded_pretraining_trainer.eval(pretraining_eval_dataloader)

    original_ce = pretraining_trainer.state.eval_metrics['eval']['LanguageCrossEntropy']
    loaded_ce = loaded_pretraining_trainer.state.eval_metrics['eval']['LanguageCrossEntropy']
    assert original_ce.compute() > 0.0
    assert original_ce.compute() == loaded_ce.compute()

    return str(tmp_path / 'pretraining_checkpoints' / 'latest-rank0.pt')


def finetuning_test_helper(tokenizer, model, algorithms, checkpoint_path, pretraining_model, tmp_path, device):
    finetuning_model_copy = copy.deepcopy(model)

    finetuning_train_dataset = RandomTextClassificationDataset(
        size=4,
        vocab_size=tokenizer.vocab_size,
        sequence_length=2,
        num_classes=3,
        use_keys=isinstance(model, HuggingFaceModel),
    )
    finetuning_train_dataloader = DataLoader(
        finetuning_train_dataset,
        batch_size=2,
        sampler=dist.get_sampler(finetuning_train_dataset),
    )
    finetuning_eval_dataloader = DataLoader(
        finetuning_train_dataset,
        batch_size=2,
        sampler=dist.get_sampler(finetuning_train_dataset),
    )

    remote_dir = str(tmp_path / 'object_store')
    os.makedirs(remote_dir, exist_ok=True)

    rud = RemoteUploaderDownloader(
        bucket_uri='libcloud://.',
        backend_kwargs={
            'provider': 'local',
            'container': '.',
            'provider_kwargs': {
                'key': remote_dir,
            },
        },
        num_concurrent_uploads=1,
        use_procs=False,
        upload_staging_folder=str(tmp_path / 'staging_folder'),
    )

    finetuning_embedding_layer = get_model_embeddings(model)
    pretraining_embedding_layer = get_model_embeddings(pretraining_model)
    # The pretraining weights have not yet been loaded into the finetuning model
    assert not torch.equal(finetuning_embedding_layer.cpu(), pretraining_embedding_layer.cpu())
    finetuning_trainer = Trainer(
        model=model,
        train_dataloader=finetuning_train_dataloader,
        save_folder='finetuning_checkpoints',
        load_path=checkpoint_path,
        load_weights_only=True,
        load_strict_model_weights=False,
        loggers=[rud],
        max_duration='2ba',
        seed=17,
        algorithms=algorithms,
        device=device,
    )
    # Now they have been loaded
    assert torch.equal(finetuning_embedding_layer.cpu(), pretraining_embedding_layer.cpu())
    finetuning_trainer.fit()
    finetuning_trainer.eval(finetuning_eval_dataloader)

    loaded_finetuning_trainer = Trainer(
        model=finetuning_model_copy,
        load_path='finetuning_checkpoints/latest-rank0.pt',
        load_object_store=rud,
        seed=17,
        algorithms=algorithms,
        device=device,
    )

    loaded_finetuning_trainer.eval(finetuning_eval_dataloader)

    original_acc = finetuning_trainer.state.eval_metrics['eval']['MulticlassAccuracy']
    loaded_acc = loaded_finetuning_trainer.state.eval_metrics['eval']['MulticlassAccuracy']
    assert original_acc.compute() > 0.0
    assert original_acc.compute() == loaded_acc.compute()

    return loaded_finetuning_trainer, finetuning_eval_dataloader, rud, 'finetuning_checkpoints/latest-rank0.pt'


def inference_test_helper(
    finetuning_output_path,
    rud,
    finetuning_model,
    algorithms,
    original_input,
    original_output,
    onnx_opset_version,
    tmp_path,
    save_format,
    device,
):
    inference_trainer = Trainer(
        model=finetuning_model,
        load_path=finetuning_output_path,
        load_weights_only=True,
        loggers=[rud],
        seed=17,
        algorithms=algorithms,
        device=device,
    )

    os.mkdir(tmp_path / 'inference_checkpoints')
    sample_input = (original_input, {})

    inference.export_for_inference(
        model=inference_trainer.state.model,
        save_format=save_format,
        save_path=str(tmp_path / 'inference_checkpoints' / f'exported_model.{save_format}'),
        sample_input=sample_input,
        onnx_opset_version=onnx_opset_version,
    )

    copied_batch = copy.deepcopy(original_input)

    if save_format == 'onnx':
        onnx = pytest.importorskip('onnx')
        ort = pytest.importorskip('onnxruntime')
        loaded_inference_model = onnx.load(str(tmp_path / 'inference_checkpoints' / 'exported_model.onnx'))
        onnx.checker.check_model(loaded_inference_model)
        ort_session = ort.InferenceSession(
            str(tmp_path / 'inference_checkpoints' / 'exported_model.onnx'),
            providers=['CPUExecutionProvider'],
        )

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
        if isinstance(original_output, torch.Tensor) else original_output.logits.detach().numpy(),
    )


@device('cpu', 'gpu')
@pytest.mark.parametrize(
    'model_type,algorithms,save_format',
    [
        ('tinybert_hf', [GatedLinearUnits], 'onnx'),
        ('simpletransformer', [], 'torchscript'),
    ],
)
@pytest.mark.parametrize('onnx_opset_version', [14, None])
def test_full_nlp_pipeline(
    model_type,
    algorithms,
    save_format,
    onnx_opset_version,
    tmp_path,
    device,
):
    """This test is intended to exercise our full pipeline for NLP.

    To this end, it performs pretraining, loads the pretrained model with a classification head for finetuning
    and finetunes it, exports the model for inference, and loads it back in to make predictions.
    """
    pytest.importorskip('libcloud')
    pytest.importorskip('transformers')

    if onnx_opset_version == None and version.parse(torch.__version__) < version.parse('1.13'):
        pytest.skip("Don't test prior PyTorch version's default Opset version.")

    algorithms = [algorithm() for algorithm in algorithms]
    device = get_device(device)
    config = None
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased', model_max_length=128)
    if model_type == 'tinybert_hf':
        # Updated minimal BERT configuration
        config = BertConfig(
            vocab_size=30522,
            hidden_size=16,
            num_hidden_layers=2,
            num_attention_heads=2,
            intermediate_size=64,
            num_labels=3,
        )
        tiny_bert_model = BertForMaskedLM(config)
        pretraining_metrics = [LanguageCrossEntropy(ignore_index=-100), MaskedAccuracy(ignore_index=-100)]
        pretraining_model = HuggingFaceModel(
            tiny_bert_model,
            tokenizer,
            use_logits=True,
            metrics=pretraining_metrics,
        )
    elif model_type == 'simpletransformer':
        pretraining_model = SimpleTransformerMaskedLM(vocab_size=30522)
    else:
        raise ValueError('Unsupported model type')
    pretraining_output_path = pretraining_test_helper(
        tokenizer,
        pretraining_model,
        algorithms,
        tmp_path,
        device,
    )

    # finetuning
    if model_type == 'tinybert_hf':
        finetuning_metric = MulticlassAccuracy(num_classes=3, average='micro')
        finetuning_model = HuggingFaceModel(
            model=BertForSequenceClassification(config),
            tokenizer=tokenizer,
            use_logits=True,
            metrics=[finetuning_metric],
        )
    elif model_type == 'simpletransformer':
        finetuning_model = SimpleTransformerClassifier(
            vocab_size=30522,
            num_classes=3,
        )
    else:
        raise ValueError('Unsupported model type.')

    finetuning_model_copy = copy.deepcopy(finetuning_model)
    finetuning_trainer, finetuning_dataloader, rud, finetuning_output_path = finetuning_test_helper(
        tokenizer,
        finetuning_model,
        algorithms,
        pretraining_output_path,
        pretraining_model,
        tmp_path,
        device,
    )

    # inference
    batch = next(iter(finetuning_dataloader))
    finetuning_trainer.state.model.to('cpu')
    finetuning_trainer.state.model.eval()
    original_output = finetuning_trainer.state.model(batch)
    inference_test_helper(
        finetuning_output_path,
        rud,
        finetuning_model_copy,
        algorithms,
        batch,
        original_output,
        onnx_opset_version,
        tmp_path,
        save_format,
        device,
    )
