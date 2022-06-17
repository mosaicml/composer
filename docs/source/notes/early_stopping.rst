|:octagonal_sign:| Early Stopping
=================================

Early stopping and threshold stopping halt training based on set criteria. In Composer, they are both implemented as callbacks that are passed to the Trainer.

.. code:: python

    trainer = Trainer(
        ...,
        callbacks=[early_stopper],
    )

Early Stopping
--------------

The EarlyStopper callback tracks a particular training or evaluation metric and stops training if the metric does not improve within a given time interval.

The EarlyStopper callback takes several parameters.

* `monitor`: The name of the metric to track.
* `dataloader_label`: The `dataloader_label` identifies which dataloader the metric belongs to. By default, the train dataloader is labeled `train`, and the evaluation dataloader is labeled `eval`, but these values can be changed.)
* `patience`: The interval of the time that the callback will wait before stopping training if the metric is not improving. You can use integers to specify the number of epochs or provide a time string -- e.g., "50ba" or "2ep" for 50 batches and 2 epochs, respectively.
* `min_delta`: If non-zero, the change in the tracked metric over the `patience` window must be at least this large.
* `comp`: A comparison operator can be provided to measure the change in the monitored metric. The comparison operator will be called like `comp(current_value, previous_best)`

.. code:: python

    from composer import Trainer
    from from composer.callbacks.early_stopper import EarlyStopper


    early_stopper = EarlyStopper("Accuracy", "my_evaluator", patience=1)

    trainer = Trainer(
        ...,
        callbacks=[early_stopper]
    )

Threshold Stopping
------------------

The ThresholdStopper callback, also monitors a specific metric, but halts training when that metric reaches a certain threshold. 

The ThresholdStopper takes the following parameters.

* `monitor`, `dataloader_label`, and `comp`: Same as the EarlyStopper callback. The comparison operator will be called `comp(current_value, threshold)`
* `threshold`: The float threshold that dictates when the halt training.
* `stop_on_batch`: If True, training will halt in the middle of a batch if the training metrics satisfy the threshold.

.. code:: python

    from composer import Trainer
    from from composer.callbacks.threshold_stopper import ThresholdStopper

    threshold_stopper = ThresholdStopper("Accuracy", "eval", threshold=0.3)

    trainer = Trainer(
        ...,
        callbacks=[threshold_stopper]
    )


Caveats
-------

* Make sure that if the ``monitor`` is in an :class:`.Evaluator`, the ``dataloader_label`` field should be set to the label of the :class:`.Evaluator`.
* When using a metric from an :class:`.Evaluator` or using a validation metric that runs on the eval :class:`Event`, then the patience interval for the :class:`EarlyStopper` should not be set to a batch timeing interval.
