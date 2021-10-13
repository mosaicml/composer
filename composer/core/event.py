# Copyright 2021 MosaicML. All Rights Reserved.

from composer.utils.string_enum import StringEnum


class Event(StringEnum):
    """Enum to represent events.

    Training Loop
    ~~~~~~~~~~~~~

    .. include:: /core/event_training_loop_event_docstring.rst


    Attributes:
        INIT: Immediately after ``model`` initialization,
            and before creation of ``optimizers`` and ``schedulers``.
            Model surgery typically occurs here.
        TRAINING_START: Start of training.
            For multi-GPU training, runs after the DDP process fork.
        EPOCH_START: Start of an epoch.
        BATCH_START: Start of a batch.
        AFTER_DATALOADER: Immediately after the dataloader is called.
            Typically used for on-GPU dataloader transforms.
        BEFORE_TRAIN_BATCH: Before the forward-loss-backward
            computation for a training batch. When using
            gradient accumulation, this is still called only once.
        BEFORE_FORWARD: Before the call to ``model.forward()``.
        AFTER_FORWARD: After the call to ``model.forward()``.
        BEFORE_LOSS: Before the call to ``model.loss()``.
        AFTER_LOSS: After the call to ``model.loss()``.
        BEFORE_BACKWARD: Before the call to ``loss.backward()``.
        AFTER_BACKWARD: After the call to ``loss.backward()``.
        AFTER_TRAIN_BATCH: After the forward-loss-backward
            computation for a training batch. When using
            gradient accumulation, this is still called only once.
        BATCH_END: End of a batch, which occurs after the optimizer step
            and any gradient scaling.
        EPOCH_END: End of an epoch.
        TRAINING_END: End of training. 

        EVAL_START: Start of evaluation through the validation dataset.
        EVAL_BATCH_START: Before the call to ``model.validate(batch)``
        EVAL_BEFORE_FORWARD: Before the call to ``model.validate(batch)``
        EVAL_AFTER_FORWARD: After the call to ``model.validate(batch)``
        EVAL_BATCH_END: After the call to ``model.validate(batch)``
        EVAL_END: End of evaluation through the validation dataset.
    """

    INIT = "init"

    TRAINING_START = "training_start"
    EPOCH_START = "epoch_start"
    BATCH_START = "batch_start"

    AFTER_DATALOADER = "after_dataloader"

    BEFORE_TRAIN_BATCH = "before_train_batch"

    BEFORE_FORWARD = "before_forward"
    AFTER_FORWARD = "after_forward"

    BEFORE_LOSS = "before_loss"
    AFTER_LOSS = "after_loss"

    BEFORE_BACKWARD = "before_backward"
    AFTER_BACKWARD = "after_backward"

    AFTER_TRAIN_BATCH = "after_train_batch"

    BATCH_END = "batch_end"
    EPOCH_END = "epoch_end"

    TRAINING_END = "training_end"

    EVAL_START = "eval_start"
    EVAL_BATCH_START = "eval_batch_start"
    EVAL_BEFORE_FORWARD = "eval_before_forward"
    EVAL_AFTER_FORWARD = "eval_after_forward"
    EVAL_BATCH_END = "eval_batch_end"
    EVAL_END = "eval_end"
