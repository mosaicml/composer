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
from torch.optim import SGD, AdamW, Optimizer
from torch.optim.optimizer import required  # type: ignore

log = logging.getLogger(__name__)

__all__ = ["DecoupledSGDW", "DecoupledAdamW", "DecoupledNVLAMB"]


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



class DecoupledNVLAMB(Optimizer):
    r"""NVIDIA-style LAMB optimizer (NVLAMB) with the weight decay term decoupled from 
    the learning rate.

    LAMB is a layerwise adaptive optimizer inspired by LARS and described in `You et
    al., (2019) <https://arxiv.org/abs/1904.00962>`_. This implementation is based on 
    https://github.com/NUS-HPC-AI-Lab/LARS-ImageNet-PyTorch/blob/main/lamb.py and 
    `NVLAMB
    <https://developer.nvidia.com/blog/pretraining-bert-with-layer-wise-adaptive-learning-rates/>`_.

    NVLAMB differs from LAMB by pre-normalizing the gradients of the entire model by their
    L2 norm, and not scaling the norm of a weight when computing the weight update.

    The standard LAMB and NVLAMB implementations couple the weight decay term with the
    learning rate. This ties the
    optimal value of :attr:`weight_decay` to :attr:`lr` and can also hurt generalization in practice. For more details on
    why decoupling might be desirable, see `Decoupled Weight Decay Regularization <https://arxiv.org/abs/1711.05101>`_.

    Arguments:
        params (list): List of parameters to update.
        lr (float, optional): Learning rate. Default: ``1e-3``.
        betas (tuple, optional): Coefficients used for computing running averages of gradient and its square
                                 Default: ``(0.9, 0.999)``.
        eps (float, optional): Term added to the denominator to improve numerical stability. Default: ``1e-8``.
        weight_decay (float, optional): Decoupled weight decay factor. Default: ``1e-2``.
        grad_pre_norm (bool, optional): Normalize gradients of entire model by their L2
            norm. Default: `True`.
        max_grad_norm (float, optional): Value used to clip global grad norm. Default: ``1.0``.
        trust_clip (bool, optional): Enable LAMBC trust ratio clipping. Default: ``False``.
    """

    @staticmethod
    def nvlamb(params: List[torch.Tensor],
               grads: List[torch.Tensor],
               exp_avgs: List[torch.Tensor],
               exp_avg_sqs: List[torch.Tensor],
               state_steps: List[int],
               *,
               beta1: float,
               beta2: float,
               lr: float,
               initial_lr: float,
               weight_decay: float,
               eps: float,
               grad_pre_norm: bool,
               max_grad_norm: float,
               trust_clip: bool) -> None:
        r"""Functional API that performs AdamW algorithm computation with decoupled weight decay.

        Args:
            params (List[torch.Tensor]): List of parameters to update.
            grads (List[torch.Tensor]): List of parameter gradients.
            exp_avgs (List[torch.Tensor]): List of average gradients.
            exp_avg_sqs (List[torch.Tensor]): List of average squared gradients.
            state_steps (Iterable[int]): List of steps taken for all parameters.
            beta1 (float): Coefficient for computing the moving average of gradient values.
            beta2 (float): Coefficient for computing the moving average of squared gradient values.
            lr (float): Learning rate.
            initial_lr (float): Initial learning rate.
            weight_decay (float): Factor for decoupled weight decay
            eps (float): Term added to the denominator to improve numerical stability.
            grad_pre_norm (bool): Normalize gradients of entire model by their L2 norm.
            max_grad_norm (float): Value used to clip global grad norm.
            trust_clip (bool): Enable LAMBC trust ratio clipping.            
        """

        # Pre-normalize gradients
        if grad_pre_norm:
            torch.nn.utils.clip_grad_norm_(grads, max_grad_norm, norm_type=2.0)

        device = params[0].device
        one_tensor = torch.tensor(1.0, device=device)  # because torch.where doesn't handle scalars correctly

        for i, param in enumerate(params):
            grad = grads[i]
            exp_avg = exp_avgs[i]
            exp_avg_sq = exp_avg_sqs[i]
            step = state_steps[i]

            bias_correction1 = 1 - beta1**step
            bias_correction2 = 1 - beta2**step

            # Decay the first and second moment running average coefficient
            exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
            denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
            # Update size
            update = (exp_avg / bias_correction1).div_(denom)

            # Used for lr scaled by trust ratio
            scaledLR = lr
            if weight_decay != 0:
                # Layer-wise LR adaptation. Skip adaptation on parameters that are
                # excluded from weight decay.

                w_norm = torch.linalg.vector_norm(param, ord=2, dim=None)
                g_norm = torch.linalg.vector_norm(update, ord=2, dim=None)

                # Compute trust ratio
                trust_ratio = torch.where(
                    w_norm > 0,
                    torch.where(g_norm > 0, w_norm / g_norm, one_tensor),
                    one_tensor,
                )
                if trust_clip:
                    # LAMBC trust clipping fixes upper bound to one
                    trust_ratio = torch.minimum(trust_ratio, one_tensor)
                
                # Scale LR by trust ratio
                scaledLR *= trust_ratio.item()

                # Perform weight decay
                decay_factor = lr / initial_lr
                param.mul_(1 - decay_factor * weight_decay)                

            param.add_(update, alpha=-scaledLR)

    def __init__(self,
                 params: List[torch.Tensor],
                 lr: float = 1e-3,
                 betas: Tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-8,
                 weight_decay: float = 1e-2,
                 grad_pre_norm: bool = False,
                 max_grad_norm: float = 1.0,
                 trust_clip: bool = False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(
                "Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(
                "Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if not 0.0 < max_grad_norm:
            raise ValueError("Invalid max_grad_norm value: {}".format(max_grad_norm))
    
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, grad_pre_norm=grad_pre_norm, max_grad_norm=max_grad_norm, trust_clip=trust_clip)
        super().__init__(params, defaults)
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
            state_steps = []
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
                    raise RuntimeError("NVLAMB does not support sparse gradients")
                grads.append(p.grad)

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avgs.append(state["exp_avg"])
                exp_avg_sqs.append(state["exp_avg_sq"])

                # update the steps for each param group update
                state["step"] += 1
                # record the step after step update
                state_steps.append(state["step"])

            self.nvlamb(params_with_grad,
                        grads,
                        exp_avgs,
                        exp_avg_sqs,
                        state_steps,
                        beta1=beta1,
                        beta2=beta2,
                        lr=lr,
                        initial_lr=initial_lr,
                        weight_decay=weight_decay,
                        eps=eps,
                        grad_pre_norm=self.defaults['grad_pre_norm'],
                        max_grad_norm=self.defaults['max_grad_norm'],
                        trust_clip=self.defaults['trust_clip'])

        return loss