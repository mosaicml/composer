# Copyright 2021 MosaicML. All Rights Reserved.

"""Optimizers with weight decay decoupled from the learning rate.

These optimizers are based off of `Decoupled Weight Decay Regularization <https://arxiv.org/abs/1711.05101>`_, which
proposes this decoupling. In general, it is recommended to use these optimizers over their native PyTorch equivalents.
"""

from __future__ import annotations

import logging
import math
from typing import List, Tuple

import torch
from torch.optim import SGD, AdamW
from torch.optim.optimizer import required  # type: ignore

log = logging.getLogger(__name__)

__all__ = ["DecoupledSGDW", "DecoupledAdamW"]


class DecoupledSGDW(SGD):
    """SGD optimizer with the weight decay term decoupled from the learning rate.

    Argument defaults are copied from :class:`torch.optim.SGD`.

    The standard `SGD <https://pytorch.org/docs/stable/generated/torch.optim.SGD.html?highlight=sgd#torch.optim.SGD>`_
    optimizer couples the weight decay term with the gradient calculation. This ties the optimal value
    of :attr:`weight_decay` to :attr:`lr` and can also hurt generalization in practice. For more details
    on why decoupling might be desirable, see `Decoupled Weight Decay Regularization <https://arxiv.org/abs/1711.05101>`_.

    Args:
        params (list): List of parameters to optimize or dicts defining parameter groups.
        lr (float, optional): Learning rate.
        momentum (int, optional): Momentum factor. Default: ``0``.
        dampening (int, optional): Dampening factor applied to the momentum. Default: ``0``.
        weight_decay (int, optional): Decoupled weight decay factor. Default: ``0``.
        nesterov (bool, optional): Enables Nesterov momentum updates. Default: ``False``.
    """

    @staticmethod
    def sgdw(params: List[torch.Tensor], d_p_list: List[torch.Tensor], momentum_buffer_list: List[torch.Tensor], *,
             weight_decay: float, momentum: float, lr: float, initial_lr: float, dampening: float, nesterov: bool):
        r"""Functional API that performs SGDW algorithm computation.

        Args:
            params (list): List of parameters to update
            d_p_list (list): List of parameter gradients
            momentum_buffer_list (list): List of momentum buffers
            weight_decay (float): Decoupled weight decay factor
            momentum (float): Momentum factor
            lr (float): Learning rate
            initial_lr (float): Initial learning rate
            dampening (float): Dampening factor for momentum update
            nesterov (bool): Enables Nesterov momentum updates
        """

        for i, param in enumerate(params):

            d_p = d_p_list[i]

            if momentum != 0:
                buf = momentum_buffer_list[i]

                if buf is None:
                    buf = torch.clone(d_p).detach()
                    momentum_buffer_list[i] = buf
                else:
                    buf.mul_(momentum).add_(d_p, alpha=1 - dampening)

                if nesterov:
                    d_p = d_p.add(buf, alpha=momentum)
                else:
                    d_p = buf

            if weight_decay != 0:
                decay_factor = lr / initial_lr
                param.mul_(1 - decay_factor * weight_decay)

            param.add_(d_p, alpha=-lr)

    def __init__(self,
                 params: List[torch.Tensor],
                 lr: float = required,
                 momentum: float = 0,
                 dampening: float = 0,
                 weight_decay: float = 0,
                 nesterov: bool = False):
        super().__init__(params, lr, momentum, dampening, weight_decay, nesterov)
        for group in self.param_groups:
            group["initial_lr"] = group["lr"]

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            d_p_list = []
            momentum_buffer_list = []
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]
            dampening = group["dampening"]
            nesterov = group["nesterov"]
            lr = group["lr"]
            initial_lr = group["initial_lr"]

            for p in group["params"]:
                if p.grad is not None:
                    params_with_grad.append(p)
                    d_p_list.append(p.grad)

                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        momentum_buffer_list.append(None)
                    else:
                        momentum_buffer_list.append(state["momentum_buffer"])

            self.sgdw(params_with_grad,
                      d_p_list,
                      momentum_buffer_list,
                      weight_decay=weight_decay,
                      momentum=momentum,
                      lr=lr,
                      initial_lr=initial_lr,
                      dampening=dampening,
                      nesterov=nesterov)

            # update momentum_buffers in state
            for p, momentum_buffer in zip(params_with_grad, momentum_buffer_list):
                state = self.state[p]
                state["momentum_buffer"] = momentum_buffer

        return loss


class DecoupledAdamW(AdamW):
    """Adam optimizer with the weight decay term decoupled from the learning rate.

    Argument defaults are copied from :class:`torch.optim.AdamW`.

    The standard `AdamW <https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html#torch.optim.AdamW>`_
    optimizer explicitly couples the weight decay term with the learning rate. This ties the
    optimal value of :attr:`weight_decay` to :attr:`lr` and can also hurt generalization in practice. For more details on
    why decoupling might be desirable, see `Decoupled Weight Decay Regularization <https://arxiv.org/abs/1711.05101>`_.

    Args:
        params (list): List of parameters to update.
        lr (float, optional): Learning rate. Default: ``1e-3``.
        betas (tuple, optional): Coefficients used for computing running averages of gradient and its square
                                 Default: ``(0.9, 0.999)``.
        eps (float, optional): Term added to the denominator to improve numerical stability. Default: ``1e-8``.
        weight_decay (float, optional): Decoupled weight decay factor. Default: ``1e-2``.
        amsgrad (bool, optional): Enables the amsgrad variant of Adam. Default: ``False``.
    """

    @staticmethod
    def adamw(params: List[torch.Tensor], grads: List[torch.Tensor], exp_avgs: List[torch.Tensor],
              exp_avg_sqs: List[torch.Tensor], max_exp_avg_sqs: List[torch.Tensor], state_steps: List[int], *,
              amsgrad: bool, beta1: float, beta2: float, lr: float, initial_lr: float, weight_decay: float,
              eps: float) -> None:
        r"""Functional API that performs AdamW algorithm computation with decoupled weight decay.

        Args:
            params (List[torch.Tensor]): List of parameters to update.
            grads (List[torch.Tensor]): List of parameter gradients.
            exp_avgs (List[torch.Tensor]): List of average gradients.
            exp_avg_sqs (List[torch.Tensor]): List of average squared gradients.
            max_exp_avg_sqs (List[torch.Tensor]): List of max average squared gradients for amsgrad updates.
            state_steps (Iterable[int]): List of steps taken for all parameters.
            amsgrad (bool): Enables amsgrad variant of Adam.
            beta1 (float): Coefficient for computing the moving average of gradient values.
            beta2 (float): Coefficient for computing the moving average of squared gradient values.
            lr (float): Learning rate.
            initial_lr (float): Initial learning rate.
            weight_decay (float): Factor for decoupled weight decay
            eps (float): Term added to the denominator to improve numerical stability.
        """

        for i, param in enumerate(params):
            grad = grads[i]
            exp_avg = exp_avgs[i]
            exp_avg_sq = exp_avg_sqs[i]
            step = state_steps[i]

            # Perform stepweight decay
            if weight_decay != 0:
                decay_factor = lr / initial_lr
                param.mul_(1 - decay_factor * weight_decay)

            bias_correction1 = 1 - beta1**step
            bias_correction2 = 1 - beta2**step

            # Decay the first and second moment running average coefficient
            exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
            if amsgrad:
                # Maintains the maximum of all 2nd moment running avg. till now
                torch.maximum(max_exp_avg_sqs[i], exp_avg_sq, out=max_exp_avg_sqs[i])
                # Use the max. for normalizing running avg. of gradient
                denom = (max_exp_avg_sqs[i].sqrt() / math.sqrt(bias_correction2)).add_(eps)
            else:
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)

            step_size = lr / bias_correction1

            param.addcdiv_(exp_avg, denom, value=-step_size)

    def __init__(self,
                 params: List[torch.Tensor],
                 lr: float = 1e-3,
                 betas: Tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-8,
                 weight_decay: float = 1e-2,
                 amsgrad: bool = False):
        super().__init__(params, lr, betas, eps, weight_decay, amsgrad)
        for group in self.param_groups:
            group["initial_lr"] = group["lr"]

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            max_exp_avg_sqs = []
            state_steps = []
            amsgrad = group["amsgrad"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            lr = group["lr"]
            initial_lr = group["initial_lr"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                params_with_grad.append(p)
                if p.grad.is_sparse:
                    raise RuntimeError("AdamW does not support sparse gradients")
                grads.append(p.grad)

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state["max_exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avgs.append(state["exp_avg"])
                exp_avg_sqs.append(state["exp_avg_sq"])

                if amsgrad:
                    max_exp_avg_sqs.append(state["max_exp_avg_sq"])

                # update the steps for each param group update
                state["step"] += 1
                # record the step after step update
                state_steps.append(state["step"])

            self.adamw(params_with_grad,
                       grads,
                       exp_avgs,
                       exp_avg_sqs,
                       max_exp_avg_sqs,
                       state_steps,
                       amsgrad=amsgrad,
                       beta1=beta1,
                       beta2=beta2,
                       lr=lr,
                       initial_lr=initial_lr,
                       weight_decay=weight_decay,
                       eps=eps)

        return loss
