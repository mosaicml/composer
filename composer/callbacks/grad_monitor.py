from dataclasses import dataclass

import yahp as hp

from composer.callbacks.callback_hparams import CallbackHparams
from composer.core import Logger, State
from composer.core.callback import Callback


class GradMonitor(Callback):

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
            state: The State object used during training.
            logger: The Logger object.
        """
        norm = None
        layer_norms = {}
        for name, p in state.model.named_parameters():
            if p.grad is not None and p.requires_grad:
                param_grad_norm = p.grad.detach().data.norm(2)
                if self.log_layer_grad_norms:
                    layer_norms[f'layer_grad_l2_norm/{name}'] = param_grad_norm

                param_grad_norm = param_grad_norm**2
                norm = param_grad_norm if not norm else norm + param_grad_norm

        norm = norm**0.5
        logger.metric_batch({'grad_l2_norm/step': norm})
        if self.log_layer_grad_norms:
            logger.metric_batch(layer_norms)


@dataclass
class GradMonitorHparams(CallbackHparams):

    log_layer_grad_norms: bool = hp.optional(
        doc="Whether or not to log gradient norms for individual layers. False by default.",
        default=False,
    )

    def initialize_object(self) -> GradMonitor:
        return GradMonitor(log_layer_grad_norms=self.log_layer_grad_norms)
