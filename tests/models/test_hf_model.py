# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0
import json
import tempfile
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader
from torchmetrics.classification import Accuracy

from composer.trainer import Trainer
from composer.utils import dist
from tests.common.datasets import RandomTextClassificationDataset


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


@pytest.mark.parametrize('pass_in_tokenizer', [True, False])
@pytest.mark.parametrize('num_classes', [2, 3])
def test_hf_state_dict_info(tmp_path: str, pass_in_tokenizer: bool, num_classes: int):
    pytest.importorskip('transformers')
    import transformers

    from composer.models import HuggingFaceModel

    config = transformers.AutoConfig.from_pretrained('prajjwal1/bert-tiny', num_labels=num_classes)
    tokenizer = transformers.AutoTokenizer.from_pretrained('prajjwal1/bert-tiny') if pass_in_tokenizer else None
    hf_model = transformers.AutoModelForSequenceClassification.from_config(config)  # type: ignore (thirdparty)

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

    expected_model_config_dict = hf_model.config.to_dict()
    new_model_config_dict = new_model_from_loaded_config.config.to_dict()
    assert expected_model_config_dict == new_model_config_dict

    if pass_in_tokenizer:
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
        assert False
    else:
        assert hf_tokenizer_state == {}
