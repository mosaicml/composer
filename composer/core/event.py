from composer.utils.string_enum import StringEnum


class Event(StringEnum):
    """
    Events, in order of execution during the training loop. The correspondig
    value to each enum here should be the event name in lowercase.
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
