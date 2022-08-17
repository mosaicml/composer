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

Multiple Datasets
-----------------

If there are multiple validation datasets that may have different metrics,
use :class:`.Evaluator` to specify each pair of dataloader and metrics.
This class is just a container for a few attributes:

- ``label``: a user-specified name for the metric.
- ``dataloader``: PyTorch :class:`~torch.utils.data.DataLoader` or our :class:`.DataSpec`.
    See :doc:`DataLoaders</trainer/dataloaders>` for more details.
- ``metric_names``: list of names of metrics to track.

For example, the `GLUE <https://gluebenchmark.com>`__ tasks for language models
can be specified as in the following example:

.. code:: python

    from composer.core import Evaluator
    from composer.models.nlp_metrics import BinaryF1Score

    glue_mrpc_task = Evaluator(
        label='glue_mrpc',
        dataloader=mrpc_dataloader,
        metric_names=['BinaryF1Score', 'Accuracy']
    )

    glue_mnli_task = Evaluator(
        label='glue_mnli',
        dataloader=mnli_dataloader,
        metric_names=['Accuracy']
    )

    trainer = Trainer(
        ...,
        eval_dataloader=[glue_mrpc_task, glue_mnli_task],
        ...
    )
