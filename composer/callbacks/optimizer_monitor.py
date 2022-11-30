# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Monitor gradients during training."""

import torch

from composer.core import Callback, State
from composer.loggers import Logger

__all__ = ['OptimizerMonitor']


class OptimizerMonitor(Callback):
    """Computes and logs the L2 norm of gradients as well as any optimizer-specific metrics implemented in the optimizer's `report_per_parameter_metrics` method.

    L2 norms are calculated after the reduction of gradients across GPUs. This function iterates over the parameters of
    the model and may cause a reduction in throughput while training large models. In order to ensure the
    correctness of the norm, this function should be called after gradient unscaling in cases where gradients are scaled.

    Example:
        .. doctest::

            >>> from composer import Trainer
            >>> from composer.callbacks import OptimizerMonitor
            >>> # constructing trainer object with this callback
            >>> trainer = Trainer(
            ...     model=model,
            ...     train_dataloader=train_dataloader,
            ...     eval_dataloader=eval_dataloader,
            ...     optimizers=optimizer,
            ...     max_duration="1ep",
            ...     callbacks=[OptimizerMonitor()],
            ... )

    The metrics are logged by the :class:`.Logger` to the following keys as described below. `grad_l2_norm` and `layer_grad_l2_norm` are
    logged in addition to metrics logged by the optimizer's `report_per_parameter_metrics` method. For convenience we have listed
    the metrics logged by DecoupledAdamW below.

    +-----------------------------------------------+-----------------------------------------------------+
    | Key                                           | Logged data                                         |
    +===============================================+=====================================================+
    |                                               | L2 norm of the gradients of all parameters in       |
    | ``l2_norm/grad/global``                       | the model on the :attr:`.Event.AFTER_TRAIN_BATCH`   |
    |                                               | event.                                              |
    +-----------------------------------------------+-----------------------------------------------------+
    |                                               | Layer-wise L2 norms                                 |
    | ``l2_norm/grad/LAYER_NAME``                   |                                                     |
    |                                               |                                                     |
    +-----------------------------------------------+-----------------------------------------------------+
    |                                               | Layer-wise L2 norms of Adam first moment after      |
    | ``l2_norm/moment/LAYER_NAME``                 |  calling optimizer step.                            |
    |                                               |                                                     |
    +-----------------------------------------------+-----------------------------------------------------+
    |                                               | Layer-wise ratio of the gradient norm to the        |
    | ``l2_norm_ratio/moment_grad/LAYER_NAME``      | moment norm after calling optimizer step.           |
    |                                               |                                                     |
    +-----------------------------------------------+-----------------------------------------------------+
    |                                               | Layer-wise cosine angle between gradient and moment |
    | ``cosine/moment_grad/LAYER_NAME``             | after calling optimizer step.                       |
    |                                               |                                                     |
    +-----------------------------------------------+-----------------------------------------------------+
    |                                               | Layer-wise L2 norms of parameter weights            |
    | ``l2_norm/param/LAYER_NAME``                  |                                                     |
    |                                               |                                                     |
    +-----------------------------------------------+-----------------------------------------------------+
    |                                               | Layer-wise L2 norms of the square root              |
    | ``l2_norm/second_moment_sqrt/LAYER_NAME``     |  of the Adam second moment is.                      |
    |                                               |                                                     |
    +-----------------------------------------------+-----------------------------------------------------+
    |                                               | Layer-wise L2 norms of the step                     |
    | ``l2_norm/update/LAYER_NAME``                 |                                                     |
    |                                               |                                                     |
    +-----------------------------------------------+-----------------------------------------------------+
    |                                               | Layer-wise cosine between the gradient and the step |
    | ``cosine/update_grad/LAYER_NAME``             |                                                     |
    |                                               |                                                     |
    +-----------------------------------------------+-----------------------------------------------------+
    |                                               | Layer-wise ratio between step size and parameter    |
    | ``l2_norm_ratio/update_param/LAYER_NAME``     | norm                                                |
    |                                               |                                                     |
    +-----------------------------------------------+-----------------------------------------------------+
    """

    def __init__(self, log_optimizer_metrics: bool = True):
        self.log_optimizer_metrics = log_optimizer_metrics

    def batch_end(self, state: State, logger: Logger):
        norm = 0.0
        default_metrics = {}
        optimizer_metrics = {}

        for name, p in state.model.named_parameters():
            if p.grad is not None and p.requires_grad:
                param_grad_norm = torch.linalg.vector_norm(p.grad)
                default_metrics[f'l2_norm/grad/{name}'] = param_grad_norm

                norm += param_grad_norm**2
                metric_reporter = getattr(state.optimizers[0], 'report_per_parameter_metrics', None)
                if callable(metric_reporter) and self.log_optimizer_metrics:
                    optimizer_metrics = metric_reporter(p, name, optimizer_metrics)

        default_metrics['l2_norm/grad/global'] = norm**0.5

        logger.log_metrics(default_metrics)
        if self.log_optimizer_metrics:
            logger.log_metrics(optimizer_metrics)
