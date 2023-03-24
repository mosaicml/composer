# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0
import copy
import json
import os
import tempfile
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import patch
from urllib.parse import urlparse

import pytest
import torch
from packaging import version
from torch.utils.data import DataLoader
from torchmetrics import Metric
from torchmetrics.classification import MulticlassAccuracy

from composer.metrics import InContextLearningLMAccuracy, LanguageCrossEntropy, MaskedAccuracy
from composer.models import HuggingFaceModel
from composer.trainer import Trainer
from composer.utils import dist, is_model_fsdp
from tests.common.datasets import RandomTextClassificationDataset, RandomTextLMDataset
from tests.common.markers import device, world_size
from tests.common.models import (configure_tiny_bert_model, configure_tiny_bert_tokenizer, configure_tiny_gpt2_model,
                                 configure_tiny_gpt2_tokenizer, configure_tiny_t5_model, configure_tiny_t5_tokenizer)
from tests.loggers.test_remote_uploader_downloader import DummyObjectStore


@pytest.mark.parametrize('num_classes', [2, 3])
def test_hf_train_eval_predict(num_classes: int, tiny_bert_config):
    transformers = pytest.importorskip('transformers')

    tiny_bert_config.num_labels = num_classes
    hf_model = transformers.AutoModelForSequenceClassification.from_config(
        tiny_bert_config)  # type: ignore (thirdparty)

    metrics = MulticlassAccuracy(num_classes=num_classes, average='micro')
    model = HuggingFaceModel(hf_model, metrics=[metrics], use_logits=True)

    vocab_size = 30522  # Match bert vocab size
    sequence_length = 4
    num_classes = num_classes
    size = 16
    batch_size = 8

    train_dataset = RandomTextClassificationDataset(size=size,
                                                    vocab_size=vocab_size,
                                                    sequence_length=sequence_length,
                                                    num_classes=num_classes,
                                                    use_keys=True)
    eval_dataset = RandomTextClassificationDataset(size=size,
                                                   vocab_size=vocab_size,
                                                   sequence_length=sequence_length,
                                                   num_classes=num_classes,
                                                   use_keys=True)
    predict_dataset = RandomTextClassificationDataset(size=size,
                                                      vocab_size=vocab_size,
                                                      sequence_length=sequence_length,
                                                      num_classes=num_classes,
                                                      use_keys=True)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=dist.get_sampler(train_dataset))
    eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, sampler=dist.get_sampler(eval_dataset))
    predict_dataloader = DataLoader(predict_dataset, batch_size=batch_size)

    trainer = Trainer(
        model=model,
        train_dataloader=train_dataloader,
        max_duration='1ep',
        eval_dataloader=eval_dataloader,
    )

    trainer.fit()
    trainer.eval()

    # Check that there is some train/eval accuracy
    assert trainer.state.train_metrics['MulticlassAccuracy'].compute() != 0.0
    assert trainer.state.eval_metrics['eval']['MulticlassAccuracy'].compute() != 0.0

    predictions = trainer.predict(predict_dataloader)

    # Check that the output predictions are the expected shape
    num_predict_batches_expected = ((size - 1) // batch_size) + 1
    assert len(predictions) == num_predict_batches_expected
    assert predictions[0]['logits'].shape == (batch_size, num_classes)


def check_hf_tokenizer_equivalence(tokenizer1, tokenizer2):
    """This is a best effort attempt to compare two tokenizers for equivalence

    This is not a perfect test, but it should catch most issues. We first check that the vocab is identical
    and that a string is tokenized the same one. Then we compare the __dict__ of the tokenizers, but we remove
    some keys that are not important for equivalence. See the inline explanations for each one.
    """
    #
    assert tokenizer1.vocab == tokenizer2.vocab
    assert type(tokenizer1) == type(tokenizer2)

    expected_tokenizer_output = tokenizer2('This is some text that should get tokenizer !? @ totallyarealtoken')
    actual_tokenizer_output = tokenizer1('This is some text that should get tokenizer !? @ totallyarealtoken')
    assert expected_tokenizer_output == actual_tokenizer_output

    # we remove the actual _tokenizer object because it is an instantiated object and so does not pass equality
    # the tokenizers are not usable below these pops
    tokenizer1.__dict__.pop('_tokenizer')
    tokenizer2.__dict__.pop('_tokenizer')

    # extra key that is not important
    tokenizer1.__dict__.pop('deprecation_warnings')
    tokenizer2.__dict__.pop('deprecation_warnings')

    # name_or_path will be the path that the tokenizer was loaded from, which will just be a temporary directory for
    # the reloaded tokenizer, so we remove it and don't compare it between the two tokenizers
    tokenizer1.__dict__.pop('name_or_path')
    tokenizer2.__dict__.pop('name_or_path')
    tokenizer1.init_kwargs.pop('name_or_path', None)
    tokenizer2.init_kwargs.pop('name_or_path', None)

    # tokenizer.init_kwargs['model_max_length'] is unset when the tokenizer does not specify it, but is set
    # to a very large number when you save and reload, so here we just check that its the same if it is present in
    # both tokenizers. There is a separate tokenizer.model_max_length that will still get checked for equivalence
    model_max_length_1 = tokenizer1.init_kwargs.get('model_max_length', None)
    model_max_length_2 = tokenizer2.init_kwargs.get('model_max_length', None)
    if model_max_length_1 is not None and model_max_length_2 is not None:
        assert model_max_length_1 == model_max_length_2

    tokenizer1.__dict__['init_kwargs'].pop('model_max_length', None)
    tokenizer2.__dict__['init_kwargs'].pop('model_max_length', None)

    # tokenizer.init_kwargs['tokenizer_file'] is unset when the tokenizer does not specify it, but is set to
    # None when you save and reload, so here we just check that its the same if it is present in both tokenizers.
    tokenizer_file_1 = tokenizer1.init_kwargs.get('tokenizer_file', None)
    tokenizer_file_2 = tokenizer2.init_kwargs.get('tokenizer_file', None)
    if tokenizer_file_1 is not None or tokenizer_file_2 is not None:
        assert tokenizer_file_1 == tokenizer_file_2

    tokenizer1.__dict__['init_kwargs'].pop('tokenizer_file', None)
    tokenizer2.__dict__['init_kwargs'].pop('tokenizer_file', None)

    assert tokenizer1.__dict__ == tokenizer2.__dict__


def check_hf_model_equivalence(model1, model2):
    expected_model_config_dict = model1.config.to_dict()
    new_model_config_dict = model2.config.to_dict()
    assert expected_model_config_dict == new_model_config_dict
    assert sum(p.numel() for p in model1.parameters()) == sum(p.numel() for p in model2.parameters())
    assert all(type(module1) == type(module2) for module1, module2 in zip(model1.modules(), model2.modules()))


@pytest.mark.parametrize('pass_in_tokenizer', [True, False])
@pytest.mark.parametrize('modify_tokenizer', [True, False])
@pytest.mark.parametrize('num_classes', [2, 3])
def test_hf_state_dict_info(tmp_path: Path, pass_in_tokenizer: bool, modify_tokenizer: bool, num_classes: int,
                            tiny_bert_tokenizer, tiny_bert_config):
    transformers = pytest.importorskip('transformers')

    if not pass_in_tokenizer and modify_tokenizer:
        pytest.skip("Invalid parametrization. Cannot modify the tokenizer if it doesn't exist.")

    tiny_bert_config.num_labels = num_classes
    tokenizer = tiny_bert_tokenizer if pass_in_tokenizer else None
    hf_model = transformers.AutoModelForSequenceClassification.from_config(
        tiny_bert_config)  # type: ignore (thirdparty)

    if modify_tokenizer:
        assert tokenizer is not None  # pyright
        tokenizer.add_special_tokens({'bos_token': '[NEWSPECIAL]'})
        tokenizer.add_special_tokens({'additional_special_tokens': ['[MOSAICML']})
        tokenizer.add_tokens(['totallyarealtoken', 'mosaicml'])
        hf_model.resize_token_embeddings(len(tokenizer))

    metrics = MulticlassAccuracy(num_classes=num_classes, average='micro')
    model = HuggingFaceModel(hf_model, tokenizer=tokenizer, metrics=[metrics], use_logits=True)

    vocab_size = 30522  # Match bert vocab size
    sequence_length = 4
    size = 4
    batch_size = 2

    train_dataset = RandomTextClassificationDataset(size=size,
                                                    vocab_size=vocab_size,
                                                    sequence_length=sequence_length,
                                                    num_classes=num_classes,
                                                    use_keys=True)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=dist.get_sampler(train_dataset))

    trainer = Trainer(model=model,
                      train_dataloader=train_dataloader,
                      max_duration='1ep',
                      save_folder=str(tmp_path),
                      save_interval='1ep',
                      save_filename='hf-checkpoint.pt')
    trainer.save_checkpoint(str(tmp_path / 'hf-checkpoint.pt'))

    loaded_checkpoint = torch.load(Path(tmp_path) / 'hf-checkpoint.pt')
    hf_state = loaded_checkpoint['state']['integrations']['huggingface']
    hf_model_state = hf_state['model']
    hf_tokenizer_state = hf_state['tokenizer']

    assert hf_model_state['config']['class'] == 'transformers.models.bert.modeling_bert.BertForSequenceClassification'

    loaded_config_dict = hf_model_state['config']['content']
    # JSON keys need to be converted back to ints, huggingface does not auto convert them along this code path
    if 'id2label' in loaded_config_dict:
        loaded_config_dict['id2label'] = {int(k): v for k, v in loaded_config_dict['id2label'].items()}

    loaded_config = transformers.AutoConfig.from_pretrained(loaded_config_dict['_name_or_path'], **loaded_config_dict)
    new_model_from_loaded_config = transformers.AutoModelForSequenceClassification.from_config(loaded_config)

    check_hf_model_equivalence(new_model_from_loaded_config, hf_model)

    if pass_in_tokenizer:
        assert tokenizer is not None  # pyright
        with tempfile.TemporaryDirectory() as _tmp_dir:
            for filename, saved_content in hf_tokenizer_state.items():
                with open(Path(_tmp_dir) / f'{filename}{saved_content["file_extension"]}', 'w') as _tmp_file:
                    if saved_content['file_extension'] == '.json':
                        json.dump(saved_content['content'], _tmp_file)
                    elif saved_content['file_extension'] == '.txt':
                        for line in saved_content['content']:
                            _tmp_file.write(line)
                            _tmp_file.write('\n')
            loaded_tokenizer = transformers.AutoTokenizer.from_pretrained(_tmp_dir)

        # for an unknown reason this key is missing when loading the saved tokenizer, but present with a value of None
        # for the original tokenizer
        loaded_tokenizer.init_kwargs['tokenizer_file'] = loaded_tokenizer.init_kwargs.get('tokenizer_file', None)

        check_hf_tokenizer_equivalence(loaded_tokenizer, tokenizer)
    else:
        assert hf_tokenizer_state == {}


def get_lm_trainer(hf_model,
                   hf_tokenizer,
                   save_folder,
                   load_path: Optional[str] = None,
                   is_conditional_generation: bool = False,
                   do_eval: bool = False,
                   fsdp_config: Optional[Dict[str, Any]] = None):
    transformers = pytest.importorskip('transformers')

    metrics: List[Metric] = [LanguageCrossEntropy(ignore_index=-100)]
    if not is_conditional_generation:
        metrics.append(MaskedAccuracy(ignore_index=-100))

    model = HuggingFaceModel(hf_model, tokenizer=hf_tokenizer, metrics=metrics, use_logits=True)

    vocab_size = hf_model.config.vocab_size
    sequence_length = 4
    size = 4
    batch_size = 4

    train_dataset = RandomTextLMDataset(size=size,
                                        vocab_size=vocab_size,
                                        sequence_length=sequence_length,
                                        use_keys=True,
                                        use_token_type_ids=not is_conditional_generation,
                                        conditional_generation=is_conditional_generation)

    if not is_conditional_generation:
        collator = transformers.DataCollatorForLanguageModeling(tokenizer=hf_tokenizer, mlm_probability=0.15)
    else:
        # Note: this could be transformers.DataCollatorForSeq2Seq(tokenizer=hf_tokenizer, model=hf_model),
        # but we want to test the scenario where the input batch does not have decoder_input_ids,
        # which DataCollatorForSeq2Seq automatically adds
        collator = transformers.DefaultDataCollator()

    train_dataloader = DataLoader(train_dataset,
                                  batch_size=batch_size,
                                  collate_fn=collator,
                                  sampler=dist.get_sampler(train_dataset))

    eval_dataloader = None
    if do_eval:
        eval_dataloader = DataLoader(train_dataset,
                                     batch_size=batch_size,
                                     collate_fn=collator,
                                     sampler=dist.get_sampler(train_dataset))

    trainer = Trainer(model=model,
                      train_dataloader=train_dataloader,
                      eval_dataloader=eval_dataloader,
                      max_duration='1ep',
                      save_folder=save_folder,
                      save_interval='1ep',
                      save_filename='hf-checkpoint.pt',
                      load_path=load_path,
                      fsdp_config=fsdp_config)
    return trainer


@pytest.mark.parametrize('pass_in_tokenizer', [True, False])
def test_hf_no_tokenizer_warning(caplog, pass_in_tokenizer: bool, tiny_bert_model, tiny_bert_tokenizer):
    pytest.importorskip('transformers')
    import logging

    with caplog.at_level(logging.WARNING, logger='composer'):
        _ = HuggingFaceModel(tiny_bert_model,
                             tokenizer=tiny_bert_tokenizer if pass_in_tokenizer else None,
                             metrics=[],
                             use_logits=True)

    if pass_in_tokenizer:
        assert len(caplog.messages) == 0
    else:
        assert caplog.messages[
            0] == 'The tokenizer was not provided. This means the tokenizer config will not be saved in the checkpoint.'


@pytest.mark.parametrize('checkpoint_upload_path', [None, 's3://checkpoints-bucket/remote-checkpoint.pt'])
@pytest.mark.parametrize('local_save_filename', [None, 'local-checkpoint.pt'])
def test_hf_loading_load_save_paths(checkpoint_upload_path: Optional[str], local_save_filename: Optional[str],
                                    tmp_path: Path, tiny_bert_model, tiny_bert_tokenizer):
    pytest.importorskip('transformers')

    trainer = get_lm_trainer(tiny_bert_model, tiny_bert_tokenizer, str(tmp_path))
    trainer.save_checkpoint(str(tmp_path / 'hf-checkpoint.pt'))

    # Just upload the checkpoint to a dummy object store outside of composer to make mocking easier
    if checkpoint_upload_path is not None:
        parsed_uri = urlparse(checkpoint_upload_path)
        object_store = DummyObjectStore(Path(parsed_uri.netloc))
        object_store.upload_object(parsed_uri.path, str(tmp_path / 'hf-checkpoint.pt'))

    checkpoint_load_path = str(tmp_path /
                               'hf-checkpoint.pt') if checkpoint_upload_path is None else checkpoint_upload_path

    local_save_checkpoint_path = None
    if local_save_filename is not None:
        local_save_checkpoint_path = str(tmp_path / 'hf-checkpoint-local.pt')

    with patch('composer.utils.file_helpers.S3ObjectStore', DummyObjectStore):
        hf_loaded_model, hf_loaded_tokenizer = HuggingFaceModel.hf_from_composer_checkpoint(
            checkpoint_path=checkpoint_load_path, local_checkpoint_save_location=local_save_checkpoint_path)

    check_hf_model_equivalence(hf_loaded_model, tiny_bert_model)
    check_hf_tokenizer_equivalence(hf_loaded_tokenizer, tiny_bert_tokenizer)

    if local_save_checkpoint_path is not None:
        assert os.path.exists(local_save_checkpoint_path)

        if checkpoint_upload_path is None:
            # the save location should be a symlink if the load path was already a local path
            assert os.path.islink(local_save_checkpoint_path)
        else:
            # just check that we ended up with an actual file, not a symlink
            assert os.path.getsize(local_save_checkpoint_path) > 1000


@pytest.mark.parametrize('modify_tokenizer', [False, True])
def test_hf_loading_tokenizer(modify_tokenizer: bool, tmp_path: Path, tiny_bert_model, tiny_bert_tokenizer):
    pytest.importorskip('transformers')

    if modify_tokenizer:
        assert tiny_bert_tokenizer is not None  # pyright
        tiny_bert_tokenizer.add_special_tokens({'bos_token': '[NEWSPECIAL]'})
        tiny_bert_tokenizer.add_special_tokens({'additional_special_tokens': ['[MOSAICML']})
        tiny_bert_tokenizer.add_tokens(['totallyarealtoken', 'mosaicml'])
        tiny_bert_model.resize_token_embeddings(len(tiny_bert_tokenizer))

    trainer = get_lm_trainer(tiny_bert_model, tiny_bert_tokenizer, str(tmp_path))
    trainer.save_checkpoint(str(tmp_path / 'hf-checkpoint.pt'))

    hf_loaded_model, hf_loaded_tokenizer = HuggingFaceModel.hf_from_composer_checkpoint(
        checkpoint_path=str(tmp_path / 'hf-checkpoint.pt'))

    check_hf_model_equivalence(hf_loaded_model, tiny_bert_model)
    check_hf_tokenizer_equivalence(hf_loaded_tokenizer, tiny_bert_tokenizer)


@pytest.mark.parametrize('num_classes', [None, 2, 3])
@pytest.mark.parametrize('model_class_name',
                         ['default', 'autoseq', 'bertseq', 'customseq', 'bertseq_string', 'autoseq_string'])
def test_hf_loading_model_classes(model_class_name: str, num_classes: Optional[int], tmp_path: Path, tiny_bert_model,
                                  tiny_bert_tokenizer):
    transformers = pytest.importorskip('transformers')

    if num_classes is not None and model_class_name not in {'autoseq', 'bertseq', 'customseq'}:
        pytest.skip('Invalid parametrization. num_classes is only for loading sequence classification models.')

    if num_classes is None and model_class_name in {'autoseq', 'bertseq', 'customseq'}:
        pytest.skip('Invalid parametrization. num_classes cannot be None for loading sequence classification models.')

    trainer = get_lm_trainer(tiny_bert_model, tiny_bert_tokenizer, str(tmp_path))
    trainer.save_checkpoint(str(tmp_path / 'hf-checkpoint.pt'))

    class CustomSequenceClassification(transformers.BertForSequenceClassification):

        def __init__(self, config):
            super().__init__(config)
            self.custom_attribute = 'mosaicml'

    model_class_name_to_class = {
        'autoseq': transformers.AutoModelForSequenceClassification,
        'bertseq': transformers.BertForSequenceClassification,
        'default': None,
        'customseq': CustomSequenceClassification,
        'bertseq_string': 'transformers.models.bert.modeling_bert.BertForSequenceClassification',
        'autoseq_string': 'transformers.AutoModelForSequenceClassification'
    }

    model_class = model_class_name_to_class[model_class_name]
    extra_model_args = {}
    if num_classes is not None:
        extra_model_args['num_labels'] = num_classes

    hf_loaded_model, hf_loaded_tokenizer = HuggingFaceModel.hf_from_composer_checkpoint(
        checkpoint_path=str(tmp_path / 'hf-checkpoint.pt'),
        model_instantiation_class=model_class,
        model_config_kwargs=extra_model_args)

    expected_model = tiny_bert_model
    if model_class_name == 'autoseq':
        config = copy.deepcopy(tiny_bert_model.config)
        config.update(extra_model_args)
        expected_model = model_class.from_config(config)
    elif model_class_name in {'bertseq', 'customseq'}:
        config = copy.deepcopy(tiny_bert_model.config)
        config.update(extra_model_args)
        expected_model = model_class(config)
    elif model_class_name == 'bertseq_string':
        config = copy.deepcopy(tiny_bert_model.config)
        config.update(extra_model_args)
        expected_model = transformers.BertForSequenceClassification(config)
    elif model_class_name == 'autoseq_string':
        config = copy.deepcopy(tiny_bert_model.config)
        config.update(extra_model_args)
        expected_model = transformers.AutoModelForSequenceClassification.from_config(config)

    if model_class_name == 'customseq':
        assert hf_loaded_model.custom_attribute == expected_model.custom_attribute

    check_hf_model_equivalence(hf_loaded_model, expected_model)
    check_hf_tokenizer_equivalence(hf_loaded_tokenizer, tiny_bert_tokenizer)


def test_hf_loading_full_model_equivalence(tmp_path: Path, tiny_bert_model, tiny_bert_tokenizer):
    pytest.importorskip('transformers')

    trainer1 = get_lm_trainer(tiny_bert_model, tiny_bert_tokenizer, str(tmp_path))
    trainer1.fit()

    hf_loaded_model, hf_loaded_tokenizer = HuggingFaceModel.hf_from_composer_checkpoint(
        checkpoint_path=str(tmp_path / 'hf-checkpoint.pt'))

    trainer2 = get_lm_trainer(hf_loaded_model,
                              hf_loaded_tokenizer,
                              str(tmp_path),
                              load_path=str(tmp_path / 'hf-checkpoint.pt'))

    # loading from the last checkpoint gets you the same model
    for p1, p2 in zip(trainer1.state.model.parameters(), trainer2.state.model.parameters()):
        torch.testing.assert_close(p1, p2)


@pytest.mark.parametrize('model_class_name', ['gpt', 'not_a_module', 'not_a_class'])
def test_hf_loading_errors(tiny_bert_model, tiny_bert_tokenizer, model_class_name, tmp_path):
    transformers = pytest.importorskip('transformers')

    trainer = get_lm_trainer(tiny_bert_model, tiny_bert_tokenizer, str(tmp_path))
    trainer.save_checkpoint(str(tmp_path / 'hf-checkpoint.pt'))

    # The compatibility of the model chosen and the model saved are up to huggingface code, but we test
    # here that one incompatible combination of BertConfig and GPT2Model errors out
    model_class_name_to_class = {
        'gpt': transformers.GPT2Model,
        'not_a_module': 'not_a_module.BertForSequenceClassification',
        'not_a_class': 'transformers.not_a_class'
    }

    error_contexts = {
        'gpt': pytest.raises(AttributeError),
        'not_a_module': pytest.raises(ValueError),
        'not_a_class': pytest.raises(ValueError)
    }
    with error_contexts[model_class_name]:
        _, _ = HuggingFaceModel.hf_from_composer_checkpoint(str(tmp_path / 'hf-checkpoint.pt'),
                                                            model_class_name_to_class[model_class_name])


@pytest.mark.parametrize('model,tokenizer', [(configure_tiny_gpt2_model, configure_tiny_gpt2_tokenizer),
                                             (configure_tiny_bert_model, configure_tiny_bert_tokenizer)])
def test_hf_auto_shift_labels(caplog, model, tokenizer):
    pytest.importorskip('transformers')

    hf_model = model()
    hf_tokenizer = tokenizer()

    # Confirm that shift_labels is automatically set to True for gpt2 and False for bert
    if hf_model.config.model_type == 'gpt':
        import logging

        hf_model.resize_token_embeddings(len(hf_tokenizer))

        with caplog.at_level(logging.WARNING, logger='composer'):
            model = HuggingFaceModel(hf_model, tokenizer=hf_tokenizer)
            assert model.shift_labels == True

        assert len(caplog.messages) == 0

        # A warning should be generated if using a Causal LM and setting shift_labels to False
        with caplog.at_level(logging.WARNING, logger='composer'):
            model = HuggingFaceModel(hf_model, tokenizer=hf_tokenizer, shift_labels=False)
            assert model.shift_labels == False

        assert caplog.messages[
            0] == 'The shift_labels argument was set to False but the model is an instance of a HuggingFace Causal LM. This may lead to incorrect behavior.'

    if hf_model.config.model_type == 'bert':
        model = HuggingFaceModel(hf_model, tokenizer=hf_tokenizer)
        assert model.shift_labels == False


def test_hf_causal_shift_labels(tiny_gpt2_model, tiny_gpt2_tokenizer):
    pytest.importorskip('transformers')

    tiny_gpt2_model.resize_token_embeddings(len(tiny_gpt2_tokenizer))
    model = HuggingFaceModel(tiny_gpt2_model, tokenizer=tiny_gpt2_tokenizer, use_logits=True)

    batch = tiny_gpt2_tokenizer('a b c d e f g h i j k', return_tensors='pt')
    batch['labels'] = batch['input_ids'].clone()

    _ = model.eval_forward(batch)
    assert isinstance(model.labels, torch.Tensor)
    assert torch.all(model.labels[..., :3] == batch['input_ids'][..., 1:4])
    assert torch.all(model.labels[..., -1] == -100)


def test_encoder_decoder(tiny_t5_model, tiny_t5_tokenizer):
    pytest.importorskip('transformers')

    trainer = get_lm_trainer(tiny_t5_model, tiny_t5_tokenizer, None, is_conditional_generation=True, do_eval=True)
    trainer.fit()
    trainer.eval()


@pytest.mark.gpu
@pytest.mark.skipif(version.parse(torch.__version__) < version.parse('1.13.0'),
                    reason='requires PyTorch 1.13 or higher')
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_hf_fsdp(tiny_bert_config, tiny_bert_tokenizer):
    transformers = pytest.importorskip('transformers')

    tiny_bert_model = transformers.AutoModelForMaskedLM.from_config(tiny_bert_config)

    fsdp_config = {
        'sharding_strategy': 'FULL_SHARD',
        'min_params': 1e8,
        'cpu_offload': False,
        'mixed_precision': 'PURE',
        'backward_prefetch': 'BACKWARD_PRE',
        'activation_checkpointing': False,
        'activation_cpu_offload': False,
        'verbose': False
    }

    trainer = get_lm_trainer(tiny_bert_model, tiny_bert_tokenizer, None, fsdp_config=fsdp_config)

    assert is_model_fsdp(trainer.state.model)

    assert trainer.state.fsdp_enabled
    trainer.fit()


def test_separate_eval_metrics(tiny_bert_model, tiny_bert_tokenizer):
    pytest.importorskip('transformers')

    hf_model = HuggingFaceModel(
        tiny_bert_model,
        tokenizer=tiny_bert_tokenizer,
        metrics=[LanguageCrossEntropy()],
        eval_metrics=[MaskedAccuracy(), InContextLearningLMAccuracy()],
    )

    assert hf_model.train_metrics is not None
    assert hf_model.val_metrics is not None
    assert hf_model.train_metrics.keys() == {'LanguageCrossEntropy'}
    assert hf_model.val_metrics.keys() == {'InContextLearningLMAccuracy', 'MaskedAccuracy'}


@pytest.mark.parametrize('checkpoint_upload_folder', [None, 's3://checkpoints-bucket/'])
@pytest.mark.parametrize('local_save_filename', [None, 'local-checkpoint.pt'])
def test_write_hf_from_composer(checkpoint_upload_folder, local_save_filename, tiny_bert_model, tiny_bert_tokenizer,
                                tmp_path):
    transformers = pytest.importorskip('transformers')

    from composer.models.huggingface import write_huggingface_pretrained_from_composer_checkpoint

    if checkpoint_upload_folder is None:
        checkpoint_upload_folder = tmp_path
    trainer = get_lm_trainer(tiny_bert_model, tiny_bert_tokenizer, str(tmp_path))
    trainer.fit()

    # Just upload to a dummy object store outside of composer to make mocking easier
    if str(checkpoint_upload_folder).startswith('s3://'):
        parsed_uri = urlparse(checkpoint_upload_folder)
        object_store = DummyObjectStore(Path(parsed_uri.netloc))
        object_store.upload_object(parsed_uri.path + 'hf-checkpoint.pt', str(tmp_path / 'hf-checkpoint.pt'))

    with patch('composer.utils.file_helpers.S3ObjectStore', DummyObjectStore):
        checkpoint_path = os.path.join(checkpoint_upload_folder, 'hf-checkpoint.pt')
        write_huggingface_pretrained_from_composer_checkpoint(checkpoint_path,
                                                              tmp_path / 'hf-save-pretrained',
                                                              local_checkpoint_save_location=local_save_filename)

    assert os.path.exists(tmp_path / 'hf-save-pretrained' / 'config.json')
    assert os.path.exists(tmp_path / 'hf-save-pretrained' / 'pytorch_model.bin')

    loaded_hf_model = transformers.AutoModelForMaskedLM.from_pretrained(tmp_path / 'hf-save-pretrained')

    # set _name_or_path so that the equivalence check passes. It is expected that these are different, because one is loaded from disk, while one is loaded from the hub
    loaded_hf_model.config._name_or_path = tiny_bert_model.config._name_or_path

    check_hf_model_equivalence(tiny_bert_model, loaded_hf_model)


@pytest.mark.parametrize('embedding_resize', ['higher', 'lower', 'no_resize'])
@pytest.mark.parametrize('allow_embedding_resizing', [True, False])
def test_embedding_resizing(tiny_bert_model, tiny_bert_tokenizer, embedding_resize, allow_embedding_resizing, caplog):
    pytest.importorskip('transformers')

    import logging

    from composer.models import HuggingFaceModel

    original_size = tiny_bert_model.config.vocab_size
    if embedding_resize == 'higher':
        tiny_bert_model.resize_token_embeddings(original_size + 100)
    elif embedding_resize == 'lower':
        tiny_bert_model.resize_token_embeddings(original_size - 100)

    error_context = pytest.raises(ValueError) if (not allow_embedding_resizing and
                                                  embedding_resize == 'lower') else nullcontext()
    with caplog.at_level(logging.WARNING, logger='composer'):
        with error_context:
            _ = HuggingFaceModel(tiny_bert_model,
                                 tokenizer=tiny_bert_tokenizer,
                                 allow_embedding_resizing=allow_embedding_resizing)
        if embedding_resize == 'lower':
            if allow_embedding_resizing:
                # when the embedding size is smaller than the tokenizer vocab size,
                # the embeddings should get resized to match the tokenizer vocab size
                assert tiny_bert_model.config.vocab_size == len(tiny_bert_tokenizer)
                assert caplog.messages[0].startswith(
                    'The number of tokens in the tokenizer is greater than the number of tokens in the model')
        elif embedding_resize == 'higher':
            # when the embedding size is greater than the tokenizer vocab size,
            # no adjustment is needed. Some embeddings will simply not be used
            assert tiny_bert_model.config.vocab_size == original_size + 100
            assert caplog.messages[0].startswith(
                'The number of tokens in the tokenizer is less than the number of tokens in the model.')
        elif embedding_resize == 'no_resize':
            assert tiny_bert_model.config.vocab_size == original_size
            assert len(caplog.messages) == 0
        else:
            raise ValueError(f'Unknown embedding_resize: {embedding_resize}')


@device('cpu', 'gpu')
@world_size(1, 2)
@pytest.mark.parametrize('use_fsdp', [True, False])
@pytest.mark.parametrize('hf_model,hf_tokenizer', [(configure_tiny_gpt2_model, configure_tiny_gpt2_tokenizer),
                                                   (configure_tiny_t5_model, configure_tiny_t5_tokenizer)])
def test_generate(device, world_size, hf_model, hf_tokenizer, use_fsdp):
    if use_fsdp and version.parse(torch.__version__) < version.parse('1.13.0'):
        pytest.skip('FSDP requires torch >= 1.13.0')

    transformers = pytest.importorskip('transformers')
    if device == 'cpu' and use_fsdp:
        pytest.skip('FSDP is not supported on CPU.')
    if world_size == 1 and use_fsdp:
        pytest.xfail((
            'Generation with world size 1 and FSDP fails with '
            '`RuntimeError: The tensor has a non-zero number of elements, '
            'but its data is not allocated yet. Caffe2 uses a lazy allocation, '
            'so you will need to call mutable_data() or raw_mutable_data() to actually allocate memory.` '
            'This issue is resolved with world size > 1 by a dummy call to forward (see HuggingFaceModel.dummy_forward_called), '
            'but for some reason fails with world size 1.'))

    fsdp_config = None
    if use_fsdp:
        fsdp_config = {
            'sharding_strategy': 'FULL_SHARD',
        }

    hf_model = hf_model()
    hf_tokenizer = hf_tokenizer()

    model = HuggingFaceModel(hf_model, tokenizer=hf_tokenizer, use_logits=True)

    # just instantiating Trainer to go through the normal FSDP code path
    trainer = Trainer(model=model, fsdp_config=fsdp_config, device=device)

    device = trainer.state.device

    if isinstance(hf_tokenizer, transformers.models.gpt2.tokenization_gpt2_fast.GPT2TokenizerFast):
        hf_tokenizer.padding_side = 'left'
    input_dict = hf_tokenizer(['hello', 'goodbyes'], return_tensors='pt', padding=True)
    for k, v in input_dict.items():
        input_dict[k] = device.tensor_to_device(v)

    generation1 = model.generate(**input_dict, max_new_tokens=5, pad_token_id=hf_tokenizer.pad_token_id)
    generation2 = model.generate(**input_dict, max_new_tokens=3, pad_token_id=hf_tokenizer.pad_token_id)

    assert generation1.shape == (2,
                                 (input_dict['input_ids'].shape[1] if not hf_model.config.is_encoder_decoder else 1) +
                                 5)
    assert generation2.shape == (2,
                                 (input_dict['input_ids'].shape[1] if not hf_model.config.is_encoder_decoder else 1) +
                                 3)

    decoded_generation1 = hf_tokenizer.batch_decode(generation1, skip_special_tokens=True)
    decoded_generation2 = hf_tokenizer.batch_decode(generation2, skip_special_tokens=True)

    assert len(decoded_generation1) == len(decoded_generation2) == 2
    assert all(isinstance(decoded_generation, str) for decoded_generation in decoded_generation1)
    assert all(isinstance(decoded_generation, str) for decoded_generation in decoded_generation2)


@device('cpu', 'gpu')
@world_size(1, 2)
@pytest.mark.parametrize('use_fsdp', [True, False])
@pytest.mark.parametrize('hf_model,hf_tokenizer', [(configure_tiny_gpt2_model, configure_tiny_gpt2_tokenizer),
                                                   (configure_tiny_t5_model, configure_tiny_t5_tokenizer)])
def test_eval_forward_generate(device, world_size, hf_model, hf_tokenizer, use_fsdp):
    if use_fsdp and version.parse(torch.__version__) < version.parse('1.13.0'):
        pytest.skip('FSDP requires torch >= 1.13.0')
    transformers = pytest.importorskip('transformers')
    if device == 'cpu' and use_fsdp:
        pytest.skip('FSDP is not supported on CPU.')
    if world_size == 1 and use_fsdp:
        pytest.xfail((
            'Generation with world size 1 and FSDP fails with '
            '`RuntimeError: The tensor has a non-zero number of elements, '
            'but its data is not allocated yet. Caffe2 uses a lazy allocation, '
            'so you will need to call mutable_data() or raw_mutable_data() to actually allocate memory.` '
            'This issue is resolved with world size > 1 by a dummy call to forward (see HuggingFaceModel.dummy_forward_called), '
            'but for some reason fails with world size 1.'))

    fsdp_config = None
    if use_fsdp:
        fsdp_config = {
            'sharding_strategy': 'FULL_SHARD',
        }

    hf_model = hf_model()
    hf_tokenizer = hf_tokenizer()

    model = HuggingFaceModel(hf_model, tokenizer=hf_tokenizer, use_logits=True)

    # just instantiating Trainer to go through the normal FSDP code path
    trainer = Trainer(model=model, fsdp_config=fsdp_config, device=device)

    device = trainer.state.device

    if isinstance(hf_tokenizer, transformers.models.gpt2.tokenization_gpt2_fast.GPT2TokenizerFast):
        hf_tokenizer.padding_side = 'left'
    input_dict = hf_tokenizer(['hello', 'goodbyes'], return_tensors='pt', padding=True)
    for k, v in input_dict.items():
        input_dict[k] = device.tensor_to_device(v)
    input_dict['mode'] = 'generate'

    input_dict['generation_length'] = 5
    input_dict['labels'] = [['answer1'], ['answer2']]
    generation1 = model.eval_forward(input_dict, None)
    input_dict['generation_length'] = 3
    input_dict['labels'] = [['answer1'], ['answer2']]
    generation2 = model.eval_forward(input_dict, None)

    assert len(generation1) == len(generation2) == 2
    assert all(isinstance(decoded_generation, str) for decoded_generation in generation1)
    assert all(isinstance(decoded_generation, str) for decoded_generation in generation2)
