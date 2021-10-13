For a conceptual understanding of when events are run within the trainer, see the below **pseudo-code** outline:

.. code-block:: python

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

.. note::

    Several events occur right after each other (e.g. ``AFTER_DATALOADER`` and ``BATCH_START``).
    We keep these separate because algorithms/callbacks may want to run,
    for example, after all the dataloader transforms.