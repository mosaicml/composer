# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0
import contextlib
import copy
import json
import os
import tempfile
from pathlib import Path
from typing import Optional
from unittest.mock import patch
from urllib.parse import urlparse

import pytest
import torch
from torch.utils.data import DataLoader
from torchmetrics.classification import Accuracy

from composer.metrics.nlp import LanguageCrossEntropy, MaskedAccuracy
from composer.trainer import Trainer
from composer.utils import dist
from tests.common.datasets import RandomTextClassificationDataset, RandomTextLMDataset
from tests.loggers.test_remote_uploader_downloader import DummyObjectStore


@pytest.mark.parametrize('num_classes', [2, 3])
def test_hf_model_forward(num_classes: int):
    transformers = pytest.importorskip('transformers')
    from transformers.modeling_outputs import SequenceClassifierOutput

    from composer.models import HuggingFaceModel

    # dummy sequence batch with 2 labels, 32 sequence length, and 30522 (bert) vocab size).
    input_ids = torch.randint(low=0, high=30522, size=(2, 32))
    labels = torch.randint(low=0, high=num_classes, size=(2,))
    token_type_ids = torch.zeros(size=(2, 32), dtype=torch.int64)
    attention_mask = torch.randint(low=0, high=1, size=(2, 32))
    batch = {
        'input_ids': input_ids,
        'labels': labels,
        'token_type_ids': token_type_ids,
        'attention_mask': attention_mask,
    }

    # non pretrained model to avoid a slow test that downloads the weights.
    config = transformers.AutoConfig.from_pretrained('bert-base-uncased', num_labels=num_classes)
    hf_model = transformers.AutoModelForSequenceClassification.from_config(config)  # type: ignore (thirdparty)
    model = HuggingFaceModel(hf_model)

    out = model(batch)
    assert isinstance(out, SequenceClassifierOutput)
    assert out.logits.shape == (2, num_classes)


@pytest.mark.parametrize('num_classes', [2, 3])
def test_hf_train_eval_predict(num_classes: int):
    transformers = pytest.importorskip('transformers')

    from composer.models import HuggingFaceModel

    config = transformers.AutoConfig.from_pretrained('prajjwal1/bert-tiny', num_labels=num_classes)
    hf_model = transformers.AutoModelForSequenceClassification.from_config(config)  # type: ignore (thirdparty)

    metrics = Accuracy()
    model = HuggingFaceModel(hf_model, metrics=[metrics], use_logits=True)

    vocab_size = 30522  # Match bert vocab size
    sequence_length = 32
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
    predict_dataloader = DataLoader(predict_dataset, batch_size=8)

    trainer = Trainer(
        model=model,
        train_dataloader=train_dataloader,
        max_duration='1ep',
        eval_dataloader=eval_dataloader,
    )

    trainer.fit()
    trainer.eval()

    # Check that there is some train/eval accuracy
    assert trainer.state.train_metrics['Accuracy'].compute() != 0.0
    assert trainer.state.eval_metrics['eval']['Accuracy'].compute() != 0.0

    predictions = trainer.predict(predict_dataloader)

    # Check that the output predictions are the expected shape
    num_predict_batches_expected = ((size - 1) // batch_size) + 1
    assert len(predictions) == num_predict_batches_expected
    assert predictions[0]['logits'].shape == (batch_size, num_classes)


def check_hf_tokenizer_equivalence(tokenizer1, tokenizer2):
    # below is a best effort attempt to compare two tokenizers for equivalence
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
def test_hf_state_dict_info(tmp_path: str, pass_in_tokenizer: bool, modify_tokenizer: bool, num_classes: int):
    transformers = pytest.importorskip('transformers')

    from composer.models import HuggingFaceModel

    if not pass_in_tokenizer and modify_tokenizer:
        pytest.skip("Invalid parametrization. Cannot modify the tokenizer if it doesn't exist.")

    config = transformers.AutoConfig.from_pretrained('prajjwal1/bert-tiny', num_labels=num_classes)
    tokenizer = transformers.AutoTokenizer.from_pretrained('prajjwal1/bert-tiny') if pass_in_tokenizer else None
    hf_model = transformers.AutoModelForSequenceClassification.from_config(config)  # type: ignore (thirdparty)

    if modify_tokenizer:
        assert tokenizer is not None  # pyright
        tokenizer.add_special_tokens({'bos_token': '[NEWSPECIAL]'})
        tokenizer.add_special_tokens({'additional_special_tokens': ['[MOSAICML']})
        tokenizer.add_tokens(['totallyarealtoken', 'mosaicml'])
        hf_model.resize_token_embeddings(len(tokenizer))

    metrics = Accuracy()
    model = HuggingFaceModel(hf_model, tokenizer=tokenizer, metrics=[metrics], use_logits=True)

    vocab_size = 30522  # Match bert vocab size
    sequence_length = 32
    size = 16
    batch_size = 8

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

    trainer.fit()

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
            # we need to set the name_or_path back because otherwise it is the tmp dir we are loading from here
            loaded_tokenizer.name_or_path = hf_tokenizer_state['tokenizer_config']['content']['name_or_path']
            loaded_tokenizer.init_kwargs['name_or_path'] = hf_tokenizer_state['tokenizer_config']['content'][
                'name_or_path']

        # for an unknown reason this key is missing when loading the saved tokenizer, but present with a value of None
        # for the original tokenizer
        loaded_tokenizer.init_kwargs['tokenizer_file'] = loaded_tokenizer.init_kwargs.get('tokenizer_file', None)

        check_hf_tokenizer_equivalence(loaded_tokenizer, tokenizer)
    else:
        assert hf_tokenizer_state == {}


@pytest.fixture()
def tiny_bert_model():
    transformers = pytest.importorskip('transformers')

    config = transformers.AutoConfig.from_pretrained('prajjwal1/bert-tiny')
    hf_model = transformers.AutoModelForMaskedLM.from_config(config)  # type: ignore (thirdparty)
    return hf_model


@pytest.fixture()
def tiny_bert_tokenizer():
    transformers = pytest.importorskip('transformers')

    hf_tokenizer = transformers.AutoTokenizer.from_pretrained('prajjwal1/bert-tiny')
    return hf_tokenizer


def get_lm_trainer(hf_model, hf_tokenizer, save_folder, load_path: Optional[str] = None):
    transformers = pytest.importorskip('transformers')
    from composer.models import HuggingFaceModel

    metrics = [
        LanguageCrossEntropy(ignore_index=-100, vocab_size=hf_model.config.vocab_size),
        MaskedAccuracy(ignore_index=-100)
    ]

    model = HuggingFaceModel(hf_model, tokenizer=hf_tokenizer, metrics=metrics, use_logits=True)

    vocab_size = 30522  # Match bert vocab size
    sequence_length = 32
    size = 16
    batch_size = 8

    train_dataset = RandomTextLMDataset(size=size,
                                        vocab_size=vocab_size,
                                        sequence_length=sequence_length,
                                        use_keys=True)

    collator = transformers.DataCollatorForLanguageModeling(tokenizer=hf_tokenizer, mlm_probability=0.15)

    train_dataloader = DataLoader(train_dataset,
                                  batch_size=batch_size,
                                  collate_fn=collator,
                                  sampler=dist.get_sampler(train_dataset))

    trainer = Trainer(model=model,
                      train_dataloader=train_dataloader,
                      max_duration='1ep',
                      save_folder=save_folder,
                      save_interval='1ep',
                      save_filename='hf-checkpoint.pt',
                      progress_bar=True,
                      load_path=load_path)
    return trainer


@pytest.mark.parametrize('pass_in_tokenizer', [True, False])
def test_hf_no_tokenizer_warning(pass_in_tokenizer: bool, tiny_bert_model, tiny_bert_tokenizer):
    pytest.importorskip('transformers')
    from composer.models import HuggingFaceModel

    warning_context = contextlib.nullcontext() if pass_in_tokenizer else pytest.warns(UserWarning)

    with warning_context:
        _ = HuggingFaceModel(tiny_bert_model,
                             tokenizer=tiny_bert_tokenizer if pass_in_tokenizer else None,
                             metrics=[],
                             use_logits=True)


@pytest.mark.parametrize('checkpoint_upload_path', [None, 's3://checkpoints-bucket/remote-checkpoint.pt'])
@pytest.mark.parametrize('local_save_filename', [None, 'local-checkpoint.pt'])
def test_hf_loading_load_save_paths(checkpoint_upload_path: Optional[str], local_save_filename: str, tmp_path: Path,
                                    tiny_bert_model, tiny_bert_tokenizer):
    pytest.importorskip('transformers')
    from composer.models import HuggingFaceModel

    trainer = get_lm_trainer(tiny_bert_model, tiny_bert_tokenizer, str(tmp_path))
    trainer.fit()

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
    from composer.models import HuggingFaceModel

    if modify_tokenizer:
        assert tiny_bert_tokenizer is not None  # pyright
        tiny_bert_tokenizer.add_special_tokens({'bos_token': '[NEWSPECIAL]'})
        tiny_bert_tokenizer.add_special_tokens({'additional_special_tokens': ['[MOSAICML']})
        tiny_bert_tokenizer.add_tokens(['totallyarealtoken', 'mosaicml'])
        tiny_bert_model.resize_token_embeddings(len(tiny_bert_tokenizer))

    trainer = get_lm_trainer(tiny_bert_model, tiny_bert_tokenizer, str(tmp_path))
    trainer.fit()

    hf_loaded_model, hf_loaded_tokenizer = HuggingFaceModel.hf_from_composer_checkpoint(
        checkpoint_path=str(tmp_path / 'hf-checkpoint.pt'))

    check_hf_model_equivalence(hf_loaded_model, tiny_bert_model)
    check_hf_tokenizer_equivalence(hf_loaded_tokenizer, tiny_bert_tokenizer)


@pytest.mark.parametrize('num_classes', [None, 2, 3])
@pytest.mark.parametrize('model_class_name', ['default', 'autoseq', 'bertseq', 'customseq', 'gpt'])
def test_hf_loading_model_classes(model_class_name: str, num_classes: Optional[int], tmp_path: Path, tiny_bert_model,
                                  tiny_bert_tokenizer):
    transformers = pytest.importorskip('transformers')

    from composer.models import HuggingFaceModel

    if num_classes is not None and model_class_name not in {'autoseq', 'bertseq', 'customseq'}:
        pytest.skip('Invalid parametrization. num_classes is only for loading sequence classification models.')

    if num_classes is None and model_class_name in {'autoseq', 'bertseq', 'customseq'}:
        pytest.skip('Invalid parametrization. num_classes cannot be None for loading sequence classification models.')

    trainer = get_lm_trainer(tiny_bert_model, tiny_bert_tokenizer, str(tmp_path))
    trainer.fit()

    class CustomSequenceClassification(transformers.BertForSequenceClassification):

        def __init__(self, config):
            super().__init__(config)
            self.custom_attribute = 'mosaicml'

    model_class_name_to_class = {
        'autoseq': transformers.AutoModelForSequenceClassification,
        'bertseq': transformers.BertForSequenceClassification,
        'default': None,
        'customseq': CustomSequenceClassification,
        'gpt': transformers.GPT2Model
    }

    model_class = model_class_name_to_class[model_class_name]
    extra_model_args = {}
    if num_classes is not None:
        extra_model_args['num_labels'] = num_classes

    # The compatibility of the model chosen and the model saved are up to huggingface code, but we test
    # here that at least one incompatible combination of BertConfig and GPT2Model errors out
    error_context = contextlib.nullcontext() if model_class_name != 'gpt' else pytest.raises(AttributeError)
    with error_context:
        hf_loaded_model, hf_loaded_tokenizer = HuggingFaceModel.hf_from_composer_checkpoint(
            checkpoint_path=str(tmp_path / 'hf-checkpoint.pt'),
            model_instantiation_class=model_class,
            model_init_kwargs=extra_model_args)

        expected_model = tiny_bert_model
        if model_class_name == 'autoseq':
            config = copy.deepcopy(tiny_bert_model.config)
            config.update(extra_model_args)
            expected_model = model_class.from_config(config)
        elif model_class_name in {'bertseq', 'customseq'}:
            config = copy.deepcopy(tiny_bert_model.config)
            config.update(extra_model_args)
            expected_model = model_class(config)

        if model_class_name == 'customseq':
            assert hf_loaded_model.custom_attribute == expected_model.custom_attribute

        check_hf_model_equivalence(hf_loaded_model, expected_model)
        check_hf_tokenizer_equivalence(hf_loaded_tokenizer, tiny_bert_tokenizer)


def test_hf_loading_full_model_equivalence(tmp_path: Path, tiny_bert_model, tiny_bert_tokenizer):
    pytest.importorskip('transformers')
    from composer.models import HuggingFaceModel

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
