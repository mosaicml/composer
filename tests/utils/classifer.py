# Copyright 2021 MosaicML. All Rights Reserved.

from typing import Optional, Sequence

import torch
import torch.nn.functional as F
from torch.optim import SGD, Optimizer
from torchmetrics import MetricCollection
from torchmetrics.classification.accuracy import Accuracy

from composer.core import Algorithm, Engine, Evaluator, Event, Precision, State
from composer.core.types import DataLoader
from composer.loggers import Logger
from tests.utils.model import SimpleModel


def _get_state(train_dataloader: DataLoader,
               eval_dataloader: DataLoader,
               rank_zero_seed: int,
               steps_per_epoch: int = 1):
    model = SimpleModel()
    steps_per_epoch = steps_per_epoch
    metric_coll = MetricCollection([Accuracy()])
    evaluators = [Evaluator(label="dummy_label", dataloader=eval_dataloader, metrics=metric_coll)]
    return State(
        model=model,
        rank_zero_seed=rank_zero_seed,
        optimizers=SGD(model.parameters(), lr=.001, momentum=0.0),
        max_duration="1ep",
        train_dataloader=train_dataloader,
        evaluators=evaluators,
        grad_accum=1,
        precision=Precision.FP32,
    )


def test_classifier_trains(
    train_dataloader: DataLoader,
    eval_dataloader: DataLoader,
    rank_zero_seed: int,
    algorithms: Sequence[Algorithm],
    n_steps: int = 1,
    check_steps: Optional[Sequence[int]] = None,
):
    if check_steps is None:
        check_steps = list(range(n_steps))

    state = _get_state(train_dataloader=train_dataloader,
                       eval_dataloader=eval_dataloader,
                       steps_per_epoch=n_steps,
                       rank_zero_seed=rank_zero_seed)
    model = state.model

    logger = Logger(state=state, destinations=[])
    engine = Engine(state=state, logger=logger)

    engine.run_event(Event.INIT)
    engine.run_event(Event.FIT_START)
    engine.run_event(Event.EPOCH_START)

    def forward(model, X, y, state):
        engine.run_event(Event.BEFORE_FORWARD)
        state.last_input = X
        state.last_target = y
        state.outputs = model(X)
        engine.run_event(Event.AFTER_FORWARD)

    def f_loss(state):
        loss = F.cross_entropy(input=state.outputs, target=state.last_target)
        assert not torch.isnan(loss)
        return loss

    opt = state.optimizers
    assert isinstance(opt, Optimizer)
    assert state.train_dataloader is not None, "train dataloader should be set"

    for step, (X, y) in enumerate(state.train_dataloader):
        # reseed here so data is same for different sets of algorithms
        torch.manual_seed(step)

        # 1st part: run one training step
        engine.run_event(Event.BATCH_START)
        engine.run_event(Event.BEFORE_TRAIN_BATCH)
        forward(model, X, y, state)
        engine.run_event(Event.BEFORE_LOSS)
        state.loss = f_loss(state)
        original_loss = state.loss
        engine.run_event(Event.AFTER_LOSS)
        engine.run_event(Event.BEFORE_BACKWARD)
        state.loss.backward()
        engine.run_event(Event.AFTER_BACKWARD)
        engine.run_event(Event.AFTER_TRAIN_BATCH)
        opt.step()
        opt.zero_grad()
        engine.run_event(Event.BATCH_END)

        # 2nd part; use same data and make sure loss went down
        if step not in check_steps:
            continue
        engine.run_event(Event.BATCH_START)
        engine.run_event(Event.BEFORE_FORWARD)
        state.outputs = model(X)
        engine.run_event(Event.AFTER_FORWARD)
        engine.run_event(Event.BEFORE_LOSS)
        new_loss = f_loss(state)
        assert not torch.isnan(new_loss)
        engine.run_event(Event.BATCH_END)

        # since we used an extremely small learning rate, the loss should
        # be lower on the same input after the update step
        assert new_loss <= original_loss + 1e-20

        # sanity check accuracy computation
        predictions = torch.argmax(state.outputs, dim=1)
        acc = (predictions == y).float().mean()
        assert not torch.any(torch.isnan(acc))
