# Copyright 2021 MosaicML. All Rights Reserved.

from composer.core import Logger, State
from composer.core.callback import Callback


class GradMonitor(Callback):
    """Logs the L2 norm.
    
    This callback logs the L2 norm of all the parameters in the model,
    and optionally, the L2 norm of the parameters by each layer.

    It logs the L2 norm on each batch under the ``grad_l2_norm/step`` key.
    If ``log_layer_grad_norms`` is True (default False), then
    layer-wise L2 norms are logged under ``layer_grad_l2_norm/LAYER_NAME``.

    Args:
        log_layer_grad_norms (bool, optional):
            Whether to log the L2 normalization of each layer.
            Defaults to False.
    """

    def __init__(self, log_layer_grad_norms: bool = False):
        super().__init__()
        self.log_layer_grad_norms = log_layer_grad_norms

    def after_train_batch(self, state: State, logger: Logger):
        """Compute the gradient L2 norm after the reduction of the
        backwards pass across GPUs. This function iterates over the
        parameters of the model and hence may cause a reduction in
        throughput while training large models. In order to ensure
        correctness, this function should be called after gradient
        unscaling in cases where gradients are scaled.

        Args:
            state (State): The :class:`~composer.core.State` object
                used during training.
            logger (Logger):
                The :class:`~composer.core.logging.logger.Logger` object.
        """
        norm = 0.0
        layer_norms = {}
        for name, p in state.model.named_parameters():
            if p.grad is not None and p.requires_grad:
                param_grad_norm = p.grad.detach().data.norm(2).item()  # type: ignore
                if self.log_layer_grad_norms:
                    layer_norms[f'layer_grad_l2_norm/{name}'] = param_grad_norm

                param_grad_norm = param_grad_norm**2
                norm += param_grad_norm

        norm = norm**0.5
        logger.metric_batch({'grad_l2_norm/step': norm})
        if self.log_layer_grad_norms:
            logger.metric_batch(layer_norms)
