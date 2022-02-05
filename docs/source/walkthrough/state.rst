composer.State
==============

.. currentmodule:: composer

The :class:`State` object is available for algorithms to modify during
:meth:`Algorithm.apply`, and captures the state of the trainer.

A summary of available attributes and properties is given below:

.. list-table::
    :widths: 20 10 70
    :header-rows: 1

    * - Attribute
      - Type
      - Description
    * - **Training arguments**
      -
      -
    * - ``model``
      - ``torch.nn.Module``
      - Model, typically as a subclass of :class:`ComposerModel`.
    * - ``train_batch_size``
      - ``int``
      - Global batch size for training
    * - ``eval_batch_size``
      - ``int``
      - Batch size for evaluation
    * - ``grad_accum``
      - ``int``
      - Gradient accumulation steps. The size of each **microbatch** would be ``train_batch_size / num_gpus / grad_accum``
    * - ``max_duration``
      - ``Time[str]``
      - Maximum number of epochs
    * - ``precision``
      - ``str | Precision``
      - Precision, one of ``[fp32, amp]``
    * - ``precision_context``
      - ``Callable``
      - Called with the precision to return a contextmanager.
    * - **Timing Information**
      -
      -
    * - ``epoch``
      - ``int``
      - The current epoch
    * - ``step``
      - ``int``
      - The current step (in terms of optimization steps)
    * - ``batch_idx``
      - ``int``
      - Index of the batch in the current epoch. Not mutable.
    * - ``steps_per_epoch``
      - ``int``
      - Number of optimization steps per epoch.
    * - **Training Loop Tensors**
      -
      -
    * - ``batch``
      - ``Batch``
      - Batch returned by the dataloader. We currently support a ``tuple`` pair of tensors, or a ``dict`` of tensors.
    * - ``batch_pair``
      - ``BatchPair``
      - Helper ``property`` that checks the batch is a tuple pair of tensors, and returns the batch.
    * - ``batch_dict``
      - ``BatchDict``
      - Helper ``property`` that checks the batch is a ``dict``, and returns the batch.
    * - ``loss``
      - ``Tensors``
      - last computed loss
    * - ``last_batch_size``
      - ``int``
      - Batch size returned from the dataloader. This can be different from the current size of ``Batch`` tensors if algorithms have modified the batch data.
    * - ``outputs``
      - ``Tensors``
      - Output of the model's forward pass. ``outputs`` is passed to the ``model.loss`` calcuation.
    * - **Optimizers**
      -
      -
    * - ``optimizers``
      - ``Optimizer | Tuple[Optimizer]``
      - Optimizers. Multiple optimizers are not currently supported.
    * - ``schedulers``
      - ``Scheduler | Tuple[Scheduler]``
      - LR schedulers, wrapped in :class:`ComposedScheduler`.
    * - ``scaler``
      - ``torch.cuda.amp.GradScaler``
      - Gradient scaler for mixed precision.
    * - **Dataloaders**
      -
      -
    * - ``train_dataloader``
      - ``DataLoader``
      - Dataloader for training.
    * - ``eval_dataloader``
      - ``DataLoader``
      - Dataloader for evaluation.
    * - **Algorithms**
      -
      -
    * - ``algorithms``
      - ``Sequence[Algorithm]``
      - List of algorithms
    * - ``callbacks``
      - ``Sequence[Callback]``
      - List of callbacks, including loggers

.. note::

    To support multi-GPU training, ``state.model`` may be wrapped in :class:`DistributedDataParallel`, and the dataloaders may be wrapped in a device-specific dataloader that handles moving tensors to device.

.. note::

    ``Schedulers`` are wrapped in ``ComposedScheduler``, which handles stepping either stepwise or epochwise, and also properly sets up learning rate warmups.


API Reference
*************

See :mod:`composer.core.state`.
