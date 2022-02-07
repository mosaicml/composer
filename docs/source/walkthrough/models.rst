composer.models
===============

Models provided to :class:`~composer.trainer.Trainer` must use the basic
interface specified by :class:`~composer.models.ComposerModel`.

Metrics and Loss Functions
--------------------------

Evaluation metrics for common tasks are
in `torchmetrics <https://torchmetrics.readthedocs.io/en/latest/references/modules.html>`_
and are directly compatible with :class:`ComposerModel`.
Additionally, we provide implementations of the following metrics and loss functions.
