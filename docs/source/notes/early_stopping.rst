|:octagonal_sign:| Early Stopping
=================================

Early stopping and threshold stopping halt training based on set criteria. In Composer, this functionality is implemented as callbacks which can be configured and passed to the Trainer.


Early Stopping
--------------

The :class:`.EarlyStopper` callback stops training if a provided metric does not improve over a certain ``patience`` window of time.

.. testcode::

    import torch
    from composer import Trainer
    from composer.callbacks.early_stopper import EarlyStopper

    early_stopper = EarlyStopper(
        monitor='Accuracy',
        dataloader_label='train',
        patience='50ba',
        comp=torch.greater,
        min_delta=0.01,
    )

    trainer = Trainer(
        model=model,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        optimizers=optimizer,
        callbacks=[early_stopper],
        max_duration="1ep",
    )

In the above example, the ``'train'`` label means the callback is tracking the ``Accuracy`` metric for the train_dataloader. The default for the evaluation dataloader is ``eval``.

We also set ``patience='50ba'`` and ``min_delta=0.01`` which means that every 50 batches, if the Accuracy does not exceed the best recorded Accuracy by ``0.01``, training is stopped. The ``comp`` argument indicates that 'better' here means higher accuracy. Note that the ``patience`` parameter can take both a time string (see :doc:`Time</trainer/time>`) or an integer which specifies a number of epochs.

For a full list of arguments, see the documentation for :class:`.EarlyStopper.`

.. note::

    When monitoring metrics from the ``eval_dataloader``, make sure that your patience is at least a few multiples of the ``eval_interval`` (e.g. if ``eval_interval='1ep'``, ``patience='4ep'``), so that the callback has a few datapoints with which to measure improvement.

Threshold Stopping
------------------

The :class:`.ThresholdStopper`` callback also monitors a specific metric, but halts training when that metric reaches a certain threshold.

.. testcode::

    from composer import Trainer
    from composer.callbacks.threshold_stopper import ThresholdStopper

    threshold_stopper = ThresholdStopper(
        monitor="Accuracy",
        dataloader_label="eval",
        threshold=0.8,
    )

    trainer = Trainer(
        model=model,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        optimizers=optimizer,
        callbacks=[threshold_stopper],
        max_duration="1ep",
    )

In this example, training will exit when the model's validation accuracy exceeds 0.8. For a full list of arguments, see the documentation for :class:`.ThresholdStopper.`

Evaluators and Multiple Metrics
-------------------------------

When there are multiple datasets and metrics to use for validation and evaluation, :class:`.Evaluator` objects can be used to pass in multiple dataloaders to the trainer. Each of the :class:`.Evaluator` objects can have multiple metrics associated. See :doc:`Evaluation</trainer/evaluation>` for more details.

Each Evaluator object is marked with a ``label`` field for logging, and a ``metrics`` field that accepts a single metric, list of metrics. These can be provided to the callbacks above to indiciate which metric to monitor.

In the example below, the callback will monitor the `Accuracy` metric in the dataloader marked `eval_dataset1`.`

.. testsetup::

    eval_dataloader2 = eval_dataloader

.. testcode::

    from composer import Trainer, Evaluator
    from torchmetrics.classification.accuracy import Accuracy
    from composer.callbacks.early_stopper import EarlyStopper

    evaluator1 = Evaluator(
        label='eval_dataset1',
        dataloader=eval_dataloader,
        metrics=Accuracy()
    )

    evaluator2 = Evaluator(
        label='eval_dataset2',
        dataloader=eval_dataloader2,
        metrics=Accuracy()
    )

    early_stopper = EarlyStopper(
        monitor='Accuracy',
        dataloader_label='eval_dataset1',
        patience=1
    )

    trainer = Trainer(
        model=model,
        train_dataloader=train_dataloader,
        eval_dataloader=[evaluator1, evaluator2],
        optimizers=optimizer,
        callbacks=[early_stopper],
        max_duration="1ep",
    )

.. note::

    When using these callbacks with :class:`.Evaluator` objects, make sure that the ``dataloader_label`` and ``label`` field match the desired :class:`.Evaluator`.
