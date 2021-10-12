# Copyright 2021 MosaicML. All Rights Reserved.

from composer.utils.string_enum import StringEnum


class Event(StringEnum):
    """An event that occurs during the execution of the training and evaluation loops.
    
    For a conceptual understanding of what events are run within the trainer, see the below *pseudo-code* outline:

    ```
    model = your_model()
    <INIT>  # model surgery here
    optimizers = SGD(model.parameters(), lr=0.01)
    schedulers = CosineAnnealing(optimizers, T_max=90)

    ddp.launch()  # for multi-GPUs, processes are forked here
    <TRAINING_START>  # has access to process rank for DDP

    for epoch in range(90):
        <EPOCH_START>

        for batch in dataloader:
            <AFTER_DATALOADER>
            <BATCH_START>

            #-- closure: forward/backward/loss -- #
            <BEFORE_TRAIN_BATCH>

            # for gradient accumulation
            for microbatch in batch:
                <BEFORE_FORWARD>
                outputs = model.forward(microbatch)
                <AFTER_FORWARD>
                <BEFORE_LOSS>
                loss = model.loss(outputs, microbatch)
                <AFTER_LOSS>
                <BEFORE_BACKWARD>
                loss.backward()
                <AFTER_BACKWARD>

            gradient_unscaling()  # for mixed precision
            gradient_clipping()
            <AFTER_TRAIN_BATCH>
            # -------------------------- #

            optimizer.step() # grad scaling (AMP) also

            <BATCH_END>
            scheduler.step('step')
            maybe_eval()

        scheduler.step('epoch')
        maybe_eval()
        <EPOCH_END>

    <TRAINING_END>

    def maybe_eval():
        <EVAL_START>

        for batch in eval_dataloader:
            <EVAL_BATCH_START>

            <EVAL_BEFORE_FORWARD>
            outputs, targets = model.validate(batch)
            <EVAL_AFTER_FORWARD>

            metrics.update(outputs, targets)
            <EVAL_BATCH_END>
        
        <EVAL_END>
    ```
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
