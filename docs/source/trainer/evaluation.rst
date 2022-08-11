|:bar_chart:| Evaluation
========================

To track training progress, validation datasets can be provided to the
Composer Trainer through the ``eval_dataloader`` parameter. The trainer
will compute evaluation metrics on the evaluation dataset at a frequency
specified by the the :class:`.Trainer` parameter ``eval_interval``.

.. code:: python

    from composer import Trainer

    trainer = Trainer(
        ...,
        eval_dataloader=my_eval_dataloader,
        eval_interval="1ep",  # Default is every epoch
    )

The metrics should be provided by :meth:`.ComposerModel.metrics`.
For more information, see the "Metrics" section in :doc:`/composer_model`.

To provide a deeper intuition, here's pseudocode for the evaluation logic that occurs every ``eval_interval``:
.. code:: python

    metrics = model.metrics(train=False)

    for batch in eval_dataloader:
        outputs, targets = model.validate(batch)
        metrics.update(outputs, targets)  # implements the torchmetrics interface

    metrics.compute()

- The trainer iterates over ``eval_dataloader`` and passes each batch to the model's :meth:`.ComposerModel.validate` method.
- Outputs of ``model.validate`` are used to update ``metrics`` (a :class:`torchmetrics.Metric` or :class:`torchmetrics.MetricCollection` returned by :meth:`.ComposerModel.metrics <model.metrics(train=False)>`).
- Finally, metrics over the whole validation dataset are computed.

Note that the tuple returned by :meth:`.ComposerModel.validate` provide the positional arguments to ``metrics.update``.
Please keep this in mind when using custom models and/or metrics.

Multiple Datasets
-----------------

If there are multiple validation datasets that may have different metrics,
use :class:`.Evaluator` to specify each pair of dataloader and metrics.
This class is just a container for a few attributes:

- ``label``: a user-specified name for the metric.
- ``dataloader``: PyTorch :class:`~torch.utils.data.DataLoader` or our :class:`.DataSpec`.
    See :doc:`DataLoaders</trainer/dataloaders>` for more details.
- ``metrics``: :class:`torchmetrics.Metric` or :class:`torchmetrics.MetricCollection`.

For example, the `GLUE <https://gluebenchmark.com>`__ tasks for language models
can be specified as in the following example:

.. code:: python

    from composer.core import Evaluator
    from torchmetrics import Accuracy, MetricCollection
    from composer.models.nlp_metrics import BinaryF1Score

    glue_mrpc_task = Evaluator(
        label='glue_mrpc',
        dataloader=mrpc_dataloader,
        metrics=MetricCollection([BinaryF1Score(), Accuracy()])
    )

    glue_mnli_task = Evaluator(
        label='glue_mnli',
        dataloader=mnli_dataloader,
        metrics=Accuracy()
    )

    trainer = Trainer(
        ...,
        eval_dataloader=[glue_mrpc_task, glue_mnli_task],
        ...
    )

In this case, the metrics from :meth:`.ComposerModel.metrics` will be ignored
since they are explicitly provided above.

Note that a single :class:`.Evaluator` can be also be passed to ``eval_dataloader``.
The evaluation logic when providing passing one or more :class:`.Evaluator` objects to ``eval_dataloader``
follows the same basic pattern illustrated above with the following differences:
- Each evaluation round, evaluation is performed separately for each evaluator provided.
- For each evaluator, outputs from ``model.validate`` are used to compute the evaluator's metrics, not the metrics provided by the model.

As before, you will need to ensure that the outputs of :meth:`.ComposerModel.validate` are consumable by ``metrics`` for each evaluator.
