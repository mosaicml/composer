|:octagonal_sign:| Early Stopping
=================================

Early stopping and threshold stopping halt training based on set criteria. In Composer, they are both implemented as callbacks that are passed to the Trainer.


Early Stopping
--------------

The EarlyStopper callback tracks a particular training or evaluation metric and stops training if the metric does not improve within a given time interval.

The EarlyStopper callback takes several parameters.

* ``monitor``: The name of the metric to track.
* ``dataloader_label``: The ``dataloader_label`` identifies which dataloader the metric belongs to. By default, the train dataloader is labeled ``train``, and the evaluation dataloader is labeled ``eval``, but these values can be changed.
* ``patience``: The interval of the time that the callback will wait before stopping training if the metric is not improving. You can use integers to specify the number of epochs or provide a time string -- e.g., "50ba" or "2ep" for 50 batches and 2 epochs, respectively.
* ``min_delta``: If non-zero, the change in the tracked metric over the ``patience`` window must be at least this large.
* ``comp``: A comparison operator can be provided to measure the change in the monitored metric. The comparison operator will be called like ``comp(current_value, previous_best)``. Defaults to :func:`torch.less` if loss, error, or perplexity are substrings of the monitored metric, otherwise defaults to :func:`torch.greater`.

.. testcode::

    import torch
    from composer import Trainer
    from from composer.callbacks.early_stopper import EarlyStopper

    early_stopper = EarlyStopper(monitor='Accuracy', dataloader_label='train', patience='50ba', comp=torch.greater, min_delta=0.01)

    trainer = Trainer(
        optimizers=optimizer,
        callbacks=[early_stopper],
        max_duration="1ep",
    )

In the above example, the ``'train'`` label means the callback is tracking the ``Accuracy`` metric for the train_dataloader (unless we changed the default label in the Trainer using the ``train_dataloader_label`` parameter).

We also set ``patience='50ba'`` and ``min_delta=0.01`` which means that every 50 batches, if the Accuracy does not exceed the best recorded Accuracy by ``0.01``, training is stopped.

Threshold Stopping
------------------

The ThresholdStopper callback, also monitors a specific metric, but halts training when that metric reaches a certain threshold.

The ThresholdStopper takes the following parameters:

* ``monitor``, ``dataloader_label``, and ``comp``: Same as the EarlyStopper callback. The comparison operator will be called ``comp(current_value, threshold)``
* ``threshold``: The float threshold that dictates when the halt training.
* ``stop_on_batch``: If True, training will halt in the middle of a batch if the training metrics satisfy the threshold.

.. testcode::

    from composer import Trainer
    from from composer.callbacks.threshold_stopper import ThresholdStopper

    threshold_stopper = ThresholdStopper(monitor="Accuracy", "eval", threshold=0.3)

    trainer = Trainer(
        train_dataloader=train_dataloader,
        optimizers=optimizer,
        callbacks=[threshold_stopper],
        max_duration="1ep",
    )

Evaluators and Multiple Metrics
-------------------------------

When there are multiple datasets and metrics to use for validation and evaluation, :class:`.Evaluator` objects can be used to
pass in multiple dataloaders to the trainer. Each of the :class:`.Evaluator` objects can take multiple metrics to be used with that dataset.
See :doc:`Evaluation</trainer/evaluation>` for more examples and a more detailed explanation.

Briefly, each Evaluator has a ``label`` field that gets used for logging, a ``metrics`` field that takes a single metric or a list of metrics, and a dataloader.

Here is an example of how to use the EarlyStopper with an Evaluator:

.. testcode::

    from composer import Trainer, Evaluator
    from torchmetrics.classification.accuracy import Accuracy
    from composer.callbacks.early_stopper import EarlyStopper

    eval_evaluator = Evaluator(label="eval_dataset1", dataloader=eval_dataloader, metrics=Accuracy())

    early_stopper = EarlyStopper(monitor='Accuracy', dataloader_label='eval_dataset1', patience=1)

    trainer = Trainer(
        train_dataloader=train_dataloader,
        eval_dataloader=eval_evaluator,
        optimizers=optimizer,
        callbacks=[early_stopper],
        max_duration="1ep",
    )

When using the EarlyStopper or ThresholdStopper callbacks with :class:`.Evaluator` objects, make sure that the ``dataloader_label`` and ``label`` field in the right :class:`.Evaluator` match.

Also make sure that when using a metric from an :class:`.Evaluator` that, the patience interval for the :class:`EarlyStopper` should be in epochs and not in batches.
