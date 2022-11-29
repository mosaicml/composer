# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from torch.utils.data import DataLoader
from torchmetrics.classification import Accuracy

from composer.trainer import Trainer
from composer.utils import dist
from tests.common.datasets import RandomTextClassificationDataset


def test_hf_model_forward():
    pytest.importorskip('transformers')
    import transformers
    from transformers.modeling_outputs import SequenceClassifierOutput

    from composer.models import HuggingFaceModel

    # dummy sequence batch with 2 labels, 32 sequence length, and 30522 (bert) vocab size).
    input_ids = torch.randint(low=0, high=30522, size=(2, 32))
    labels = torch.randint(low=0, high=1, size=(2,))
    token_type_ids = torch.zeros(size=(2, 32), dtype=torch.int64)
    attention_mask = torch.randint(low=0, high=1, size=(2, 32))
    batch = {
        'input_ids': input_ids,
        'labels': labels,
        'token_type_ids': token_type_ids,
        'attention_mask': attention_mask,
    }

    # non pretrained model to avoid a slow test that downloads the weights.
    config = transformers.AutoConfig.from_pretrained('bert-base-uncased', num_labels=2)
    hf_model = transformers.AutoModelForSequenceClassification.from_config(config)  # type: ignore (thirdparty)
    model = HuggingFaceModel(hf_model)

    out = model(batch)
    assert isinstance(out, SequenceClassifierOutput)
    assert out.logits.shape == (2, 2)


def test_hf_train_eval_predict():
    pytest.importorskip('transformers')
    import transformers

    from composer.models import HuggingFaceModel

    config = transformers.AutoConfig.from_pretrained('prajjwal1/bert-tiny', num_labels=2)
    hf_model = transformers.AutoModelForSequenceClassification.from_config(config)  # type: ignore (thirdparty)

    metrics = Accuracy()
    model = HuggingFaceModel(hf_model, metrics=[metrics], use_logits=True)

    vocab_size = 30522  # Match bert vocab size
    sequence_length = 32
    num_classes = 2
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
    assert predictions[0]['logits'].shape == (batch_size, 2)
