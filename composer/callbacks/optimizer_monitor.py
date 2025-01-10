# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Monitor gradients during training."""

import warnings
from typing import Union

import torch

from composer.core import Callback, State, Time, TimeUnit
from composer.loggers import Logger
from composer.utils import dist

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
    |                                               | Layer-wise L2 norms of parameter weights            |
    | ``l2_norm/param/LAYER_NAME``                  |                                                     |
    |                                               |                                                     |
    +-----------------------------------------------+-----------------------------------------------------+
    |                                               | Layer-wise L2 norms of the step                     |
    | ``l2_norm/update/LAYER_NAME``                 |                                                     |
    |                                               |                                                     |
    +-----------------------------------------------+-----------------------------------------------------+
    """

    def __init__(
        self,
        only_global: bool = False,
        log_optimizer_metrics: bool = True,
        interval: Union[int, str, Time] = '10ba',
    ):
        self.log_optimizer_metrics = log_optimizer_metrics
        self.only_global = only_global

        # Check that the interval timestring is parsable and convert into time object
        if isinstance(interval, int):
            self.interval = Time(interval, TimeUnit.BATCH)
        elif isinstance(interval, str):
            self.interval = Time.from_timestring(interval)
        elif isinstance(interval, Time):
            self.interval = interval

        if self.interval.unit == TimeUnit.BATCH and self.interval < Time.from_timestring('10ba'):
            warnings.warn(
                f'Currently the ActivationMonitor`s interval is set to {self.interval} '
                f'which is below our recommended value of 10ba. We recommend you raise '
                f'the interval to at least 10ba, as the activation monitor adds extra overhead '
                f'and decreases throughput.',
            )

        # Verify that the interval has supported units
        if self.interval.unit not in [TimeUnit.BATCH, TimeUnit.EPOCH]:
            raise ValueError(f'Invalid time unit for parameter interval: '
                             f'{self.interval.unit}')

    def batch_end(self, state: State, logger: Logger):
        current_time_value = state.timestamp.get(self.interval.unit).value

        if current_time_value % self.interval.value != 0:
            return

        optimizer_metrics = {}

        for name, p in state.model.named_parameters():
            if p.grad is not None and p.requires_grad:

                metric_reporter = getattr(state.optimizers[0], 'report_per_parameter_metrics', None)
                if callable(metric_reporter) and self.log_optimizer_metrics:
                    optimizer_metrics.update(metric_reporter(p, name, optimizer_metrics))

                # Always log grad norm as a default metric if it's not specified
                if f'l2_norm/grad/{name}' not in optimizer_metrics:
                    param_grad_norm = torch.linalg.vector_norm(p.grad)
                    optimizer_metrics[f'l2_norm/grad/{name}'] = param_grad_norm

        if state.fsdp_enabled and dist.get_world_size() > 0 and self.log_optimizer_metrics:
            # If FSDP is enabled, the optimizer state lives on different ranks and must be reduced
            # and combined before we can compute metrics.
            # Each metric has a different way of being reduced, so the optimizer is responsible for implementing
            # the reduction process.
            # It occurs first via a pre-reduce, where the metric on each rank is modified and prepared
            # then an all-reduce where the modified metric on each rank is combined into the correct metric across all ranks.
            #
            # For example, L2 norms are squared on each rank before we apply all_reduce(SUM) and take the sqrt on each rank
            pre_reduce_metrics = getattr(state.optimizers[0], 'pre_reduce_metrics', None)
            if callable(pre_reduce_metrics) and self.log_optimizer_metrics:
                optimizer_metrics = pre_reduce_metrics(optimizer_metrics)

            dist_reduce_metrics = getattr(state.optimizers[0], 'dist_reduce_metrics', None)
            if callable(dist_reduce_metrics) and self.log_optimizer_metrics:
                optimizer_metrics = dist_reduce_metrics(optimizer_metrics)

        grad_norm, moment_norm, update_norm, param_norm = .0, .0, .0, .0
        for metric in optimizer_metrics:
            if metric.startswith('l2_norm/grad'):
                grad_norm += optimizer_metrics[metric]**2
            if metric.startswith('l2_norm/moment'):
                moment_norm += optimizer_metrics[metric]**2
            if metric.startswith('l2_norm/update'):
                update_norm += optimizer_metrics[metric]**2
            if metric.startswith('l2_norm/param'):
                param_norm += optimizer_metrics[metric]**2

        if self.only_global:
            optimizer_metrics = {}

        optimizer_metrics['l2_norm/grad/global'] = grad_norm**0.5
        optimizer_metrics['l2_norm/moment/global'] = moment_norm**0.5
        optimizer_metrics['l2_norm/update/global'] = update_norm**0.5
        optimizer_metrics['l2_norm/param/global'] = param_norm**0.5

        for metric in optimizer_metrics:
            if isinstance(optimizer_metrics[metric], torch.Tensor):
                optimizer_metrics[metric] = optimizer_metrics[metric].item()
        logger.log_metrics(optimizer_metrics)
