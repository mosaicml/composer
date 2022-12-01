# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Monitor gradients during training."""

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
    | ``grad_l2_norm/step``                         | the model on the :attr:`.Event.AFTER_TRAIN_BATCH`   |
    |                                               | event.                                              |
    +-----------------------------------------------+-----------------------------------------------------+
    |                                               | Layer-wise L2 norms of ``log_layer_grad_norms``     |
    | ``layer_grad_l2_norm/LAYER_NAME``             | is ``True``. Default: ``False``.                    |
    |                                               |                                                     |
    +-----------------------------------------------+-----------------------------------------------------+
    |                                               | Layer-wise L2 norms of Adam first moment after      |
    | ``layer_moment_l2_norm/LAYER_NAME``           |  calling optimizer step.                            |
    |                                               |                                                     |
    +-----------------------------------------------+-----------------------------------------------------+
    |                                               | Layer-wise ratio of the gradient norm to the        |
    | ``layer_moment_grad_norm_ratio/LAYER_NAME``   | moment norm after calling optimizer step.           |
    |                                               |                                                     |
    +-----------------------------------------------+-----------------------------------------------------+
    |                                               | Layer-wise cosine angle between gradient and moment |
    | ``layer_moment_grad_cosine/LAYER_NAME``       | after calling optimizer step.                       |
    |                                               |                                                     |
    +-----------------------------------------------+-----------------------------------------------------+
    |                                               | Layer-wise L2 norms of parameter weights            |
    | ``layer_param_norm/LAYER_NAME``               |                                                     |
    |                                               |                                                     |
    +-----------------------------------------------+-----------------------------------------------------+
    |                                               | Layer-wise L2 norms of Adam second moment           |
    | ``layer_second_moment_l2_norm/LAYER_NAME``    | is ``True``. Default: ``False``.                    |
    |                                               |                                                     |
    +-----------------------------------------------+-----------------------------------------------------+
    |                                               | Layer-wise L2 norms of the step                     |
    | ``layer_step_norm/LAYER_NAME``                |                                                     |
    |                                               |                                                     |
    +-----------------------------------------------+-----------------------------------------------------+
    |                                               | Layer-wise cosine between the gradient and the step |
    | ``layer_step_grad_cosine/LAYER_NAME``         |                                                     |
    |                                               |                                                     |
    +-----------------------------------------------+-----------------------------------------------------+
    |                                               | Layer-wise ratio between step size and parameter    |
    | ``layer_step_param_norm_ratio/LAYER_NAME``    | norm                                                |
    |                                               |                                                     |
    +-----------------------------------------------+-----------------------------------------------------+

    Args:
        log_layer_grad_norms (bool, optional): Whether to log the L2 normalization of each layer.
            Default: ``False``.
        log_layer_grad_norms (bool, optional): Whether to log optimizer-specific metrics.
            Default: ``False``.
    """

    def __init__(self, log_layer_grad_norms: bool = False, log_optimizer_metrics: bool = False):
        self.log_layer_grad_norms = log_layer_grad_norms
        self.log_optimizer_metrics = log_optimizer_metrics

    def batch_end(self, state: State, logger: Logger):
        norm = 0.0
        layer_norms = {}
        optimizer_metrics = {}

        for name, p in state.model.named_parameters():
            if p.grad is not None and p.requires_grad:
                param_grad_norm = p.grad.detach().data.norm(2).item()  # type: ignore
                if self.log_layer_grad_norms:
                    layer_norms[f'layer_grad_l2_norm/{name}'] = param_grad_norm

                param_grad_norm = param_grad_norm**2
                norm += param_grad_norm
                metric_reporter = getattr(state.optimizers[0], 'report_per_parameter_metrics', None)
                if callable(metric_reporter) and self.log_optimizer_metrics:
                    optimizer_metrics = metric_reporter(p, name, optimizer_metrics)

        norm = norm**0.5
        logger.log_metrics({'grad_l2_norm/step': norm})
        if self.log_layer_grad_norms:
            logger.log_metrics(layer_norms)

        if self.log_optimizer_metrics:
            logger.log_metrics(optimizer_metrics)
