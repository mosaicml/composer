
.. list-table::
    :widths: 25 75
    :header-rows: 1

    * - Name
      - Description
    * - ``INIT``
      - Invoked in :meth:`Trainer.__init__`. Model surgery typically occurs here.
    * - ``FIT_START``
      - Invoked at the beginning of every call to :meth:`Trainer.fit`. Dataset transformations typically occur here.
    * - ``EPOCH_START``, ``EPOCH_END``
      - Start and end of an Epoch.
    * - ``BATCH_START``, ``BATCH_END``
      - Start and end of a batch, inclusive of the optimizer step and any gradient scaling.
    * - ``AFTER_DATALOADER``
      - Immediately after the dataloader is called. Typically used for on-GPU dataloader transforms.
    * - ``BEFORE_TRAIN_BATCH``, ``AFTER_TRAIN_BATCH``
      - Before and after the forward-loss-backward computation for a training batch. When using gradient_accumulation, these are still called only once.
    * - ``BEFORE_FORWARD``, ``AFTER_FORWARD``
      - Before and after the call to ``model.forward()``
    * - ``BEFORE_LOSS``, ``AFTER_LOSS``
      - Before and after the loss computation.
    * - ``BEFORE_BACKWARD``, ``AFTER_BACKWARD``
      - Before and after the backward pass.
    * - ``EVAL_START``, ``EVAL_END``
      - Start and end of evaluation through the validation dataset.
    * - ``EVAL_BATCH_START``, ``EVAL_BATCH_END``
      - Before and after the call to ``model.validate(batch)``
    * - ``EVAL_BEFORE_FORWARD``, ``EVAL_AFTER_FORWARD``
      - Before and after the call to ``model.validate(batch)``
