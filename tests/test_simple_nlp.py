# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

from torch.utils.data import DataLoader

from composer.trainer import Trainer
from composer.utils import dist
from tests.common.datasets import RandomTextClassificationDataset
from tests.common.models import SimpleTransformerClassifier


def test_simple_nlp():
    vocab_size = 100
    sequence_length = 32
    num_classes = 2
    size = 100
    batch_size = 8

    train_dataset = RandomTextClassificationDataset(size=size,
                                                    vocab_size=vocab_size,
                                                    sequence_length=sequence_length,
                                                    num_classes=num_classes)
    eval_dataset = RandomTextClassificationDataset(size=size,
                                                   vocab_size=vocab_size,
                                                   sequence_length=sequence_length,
                                                   num_classes=num_classes)
    predict_dataset = RandomTextClassificationDataset(size=size,
                                                      vocab_size=vocab_size,
                                                      sequence_length=sequence_length,
                                                      num_classes=num_classes)

    model = SimpleTransformerClassifier(vocab_size=vocab_size, num_classes=num_classes)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=dist.get_sampler(train_dataset))
    eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, sampler=dist.get_sampler(eval_dataset))
    predict_dataloader = DataLoader(predict_dataset, batch_size=8)

    trainer = Trainer(
        model=model,
        train_dataloader=train_dataloader,
        max_duration='2ep',
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
    assert predictions[0].shape == (batch_size, 2)
