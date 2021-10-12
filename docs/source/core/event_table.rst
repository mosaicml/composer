
.. list-table::
    :widths: 25 75
    :header-rows: 1

    * - Name
      - Description
    * - ``INIT``
      - Immediately after ``model`` initialization, and before creation of ``optimizers`` and ``schedulers``. Model surgery typically occurs here.
    * - ``TRAINING_START``
      - Start of training. For multi-GPU training, runs after the DDP process fork.
    * - ``EPOCH_START``, ``EPOCH_END``
      - Start and end of an Epoch.
    * - ``BATCH_START``, ``BATCH_END``
      - Start and end of a batch, inclusive of the optimizer step and any gradient scaling.
    * - ``AFTER_DATALOADER``
      - Immediately after the dataloader is called. Typically used for on-GPU dataloader trainsforms.
    * - ``BEFORE_TRAIN_BATCH``, ``AFTER_TRAIN_BATCH``
      - Before and after the forward-loss-backward computation for a training batch. When using gradient_accumulation, these are still called only once.
    * - ``BEFORE_FORWARD``, ``AFTER_FORWARD``
      - Before and after the call to ``model.forward()``
    * - ``BEFORE_LOSS``, ``AFTER_LOSS``
      - Before and after the loss computation.
    * - ``BEFORE_BACKWARD``, ``AFTER_BACKWARD``
      - Before and after the backward pass.
    * - ``TRAINING_END``
      - End of training.