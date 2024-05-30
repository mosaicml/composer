# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Optimizers with weight decay decoupled from the learning rate.

These optimizers are based off of `Decoupled Weight Decay Regularization <https://arxiv.org/abs/1711.05101>`_, which
proposes this decoupling. In general, it is recommended to use these optimizers over their native PyTorch equivalents.
"""

from __future__ import annotations

import logging
import math
from typing import Iterable, Optional, Union

import torch
from torch.optim import SGD, AdamW
from torch.optim.optimizer import required  # type: ignore

from composer.utils import dist

log = logging.getLogger(__name__)

__all__ = ['DecoupledSGDW', 'DecoupledAdamW']


class DecoupledSGDW(SGD):
    """SGD optimizer with the weight decay term decoupled from the learning rate.

    NOTE: Since `weight_decay` is no longer scaled by `lr`, you will likely want to use much smaller values
    for `weight_decay` than you would if using `torch.optim.SGD`. In this optimizer, the value `weight_decay` translates exactly to:
    'On every optimizer update, every weight element will be multiplied by `(1.0 - weight_decay_t)`'.
    The term `weight_decay_t` will follow the same schedule as `lr_t` but crucially will not be scaled by `lr`.

    Argument defaults are copied from :class:`torch.optim.SGD`.

    Why use this optimizer? The standard `SGD <https://pytorch.org/docs/stable/generated/torch.optim.SGD.html?highlight=sgd#torch.optim.SGD>`_
    optimizer couples the weight decay term with the gradient calculation. This ties the optimal value
    of :attr:`weight_decay` to :attr:`lr` and can also hurt generalization in practice. For more details
    on why decoupling might be desirable, see `Decoupled Weight Decay Regularization <https://arxiv.org/abs/1711.05101>`_.

    Args:
        params (iterable): Iterable of parameters to optimize or dicts defining parameter groups.
        lr (float): Learning rate.
        momentum (int, optional): Momentum factor. Default: ``0``.
        dampening (int, optional): Dampening factor applied to the momentum. Default: ``0``.
        weight_decay (int, optional): Decoupled weight decay factor. Default: ``0``.
        nesterov (bool, optional): Enables Nesterov momentum updates. Default: ``False``.
    """

    def __init__(
        self,
        params: Union[Iterable[torch.Tensor], Iterable[dict]],
        lr: float = required,  # type: ignore
        momentum: float = 0,
        dampening: float = 0,
        weight_decay: float = 0,
        nesterov: bool = False,
    ):
        if weight_decay >= 1e-3:
            log.warning(
                f'You are using a high value of `weight_decay={weight_decay}` for the `DecoupledSGDW` optimizer. Are you sure you want to do this? '
                f'Your model\'s weights will be multiplied by {1.0 - weight_decay} on every step!',
            )
        super().__init__(
            params=params,
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
        )
        for group in self.param_groups:
            group['initial_lr'] = group['lr']

    @staticmethod
    def sgdw(
        params: list[torch.Tensor],
        d_p_list: list[torch.Tensor],
        momentum_buffer_list: list[Optional[torch.Tensor]],
        *,
        weight_decay: float,
        momentum: float,
        lr: float,
        initial_lr: float,
        dampening: float,
        nesterov: bool,
    ):
        r"""Functional API that performs SGDW algorithm computation.

        Args:
            params (list): list of parameters to update
            d_p_list (list): list of parameter gradients
            momentum_buffer_list (list): list of momentum buffers
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
                decay_factor = (lr / initial_lr) if initial_lr else 1.0
                param.mul_(1 - decay_factor * weight_decay)

            param.add_(d_p, alpha=-lr)

    @torch.no_grad()  # pyright: ignore[reportUntypedFunctionDecorator]
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
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            lr = group['lr']
            initial_lr = group['initial_lr']

            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    d_p_list.append(p.grad)

                    state = self.state[p]
                    if 'momentum_buffer' not in state:
                        momentum_buffer_list.append(None)
                    else:
                        momentum_buffer_list.append(state['momentum_buffer'])

            self.sgdw(
                params_with_grad,
                d_p_list,
                momentum_buffer_list,
                weight_decay=weight_decay,
                momentum=momentum,
                lr=lr,
                initial_lr=initial_lr,
                dampening=dampening,
                nesterov=nesterov,
            )

            # update momentum_buffers in state
            for p, momentum_buffer in zip(params_with_grad, momentum_buffer_list):
                state = self.state[p]
                state['momentum_buffer'] = momentum_buffer

        return loss


class DecoupledAdamW(AdamW):
    """Adam optimizer with the weight decay term decoupled from the learning rate.

    NOTE: Since `weight_decay` is no longer scaled by `lr`, you will likely want to use much smaller values
    for `weight_decay` than you would if using `torch.optim.Adam` or `torch.optim.AdamW`. In this optimizer, the value `weight_decay` translates exactly to:
    'On every optimizer update, every weight element will be multiplied by `(1.0 - weight_decay_t)`'.
    The term `weight_decay_t` will follow the same schedule as `lr_t` but crucially will not be scaled by `lr`.

    Argument defaults are similar to :class:`torch.optim.AdamW` but we make two changes:
    * The default for ``weight_decay`` is changed from ``1e-2`` -> ``1e-5`` because in `DecoupledAdamW`, the weight decay is decoupled and no longer scaled by the `lr=1e-3`.
    * The default for ``betas`` is changed from ``(0.9, 0.999)`` to ``(0.9, 0.95)`` to reflect community best-practices for the beta2 hyperparameter.

    Why use this optimizer? The standard `AdamW <https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html#torch.optim.AdamW>`_
    optimizer explicitly couples the weight decay term with the learning rate. This ties the
    optimal value of :attr:`weight_decay` to :attr:`lr` and can also hurt generalization in practice. For more details on
    why decoupling might be desirable, see `Decoupled Weight Decay Regularization <https://arxiv.org/abs/1711.05101>`_.

    Args:
        params (iterable): Iterable of parameters to optimize or dicts defining parameter groups.
        lr (float, optional): Learning rate. Default: ``1e-3``.
        betas (tuple, optional): Coefficients used for computing running averages of gradient and its square
                                 Default: ``(0.9, 0.95)``.
        eps (float, optional): Term added to the denominator to improve numerical stability. Default: ``1e-8``.
        weight_decay (float, optional): Decoupled weight decay factor. Default: ``1e-5``.
        amsgrad (bool, optional): Enables the amsgrad variant of Adam. Default: ``False``.
    """

    metric_functions = {
        'l2_norm/moment': lambda param, optim_state, step_tensor: torch.linalg.vector_norm(optim_state['exp_avg']),
        'l2_norm/param': lambda param, optim_state, step_tensor: torch.linalg.vector_norm(param.data),
        'l2_norm/update': lambda param, optim_state, step_tensor: torch.linalg.vector_norm(step_tensor),
        'l2_norm/grad': lambda param, optim_state, step_tensor: torch.linalg.vector_norm(param.grad),
    }

    def __init__(
        self,
        params: Union[Iterable[torch.Tensor], Iterable[dict]],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
        weight_decay: float = 1e-5,
        amsgrad: bool = False,
    ):
        if weight_decay >= 1e-3:
            log.warning(
                f'You are using a high value of `weight_decay={weight_decay}` for the `DecoupledAdamW` optimizer. Are you sure you want to do this? '
                f'Your model\'s weights will be multiplied by {1.0 - weight_decay} on every step!',
            )
        super().__init__(params=params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad)
        for group in self.param_groups:
            group['initial_lr'] = group['lr']
        self.amsgrad = amsgrad

    @staticmethod
    def adamw(
        params: list[torch.Tensor],
        grads: list[torch.Tensor],
        exp_avgs: list[torch.Tensor],
        exp_avg_sqs: list[torch.Tensor],
        max_exp_avg_sqs: list[torch.Tensor],
        state_steps: list[torch.Tensor],
        *,
        amsgrad: bool,
        beta1: float,
        beta2: float,
        lr: float,
        initial_lr: float,
        weight_decay: float,
        eps: float,
    ) -> None:
        r"""Functional API that performs AdamW algorithm computation with decoupled weight decay.

        Args:
            params (list): list of parameters to update.
            grads (list): list of parameter gradients.
            exp_avgs (list): list of average gradients.
            exp_avg_sqs (list): list of average squared gradients.
            max_exp_avg_sqs (list): list of max average squared gradients for amsgrad updates.
            state_steps (list): list of steps taken for all parameters.
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
            step = state_steps[i].item()

            # Perform stepweight decay
            if weight_decay != 0:
                decay_factor = (lr / initial_lr) if initial_lr else 1.0
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

    @torch.no_grad()  # pyright: ignore[reportUntypedFunctionDecorator]
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
            amsgrad = group['amsgrad']
            beta1, beta2 = group['betas']
            eps = group['eps']
            lr = group['lr']
            if 'initial_lr' not in group:
                group['initial_lr'] = lr
            initial_lr = group['initial_lr']
            weight_decay = group['weight_decay']

            for p in group['params']:
                if p.grad is None or not p.requires_grad:
                    continue
                params_with_grad.append(p)
                if p.grad.is_sparse:
                    raise RuntimeError('AdamW does not support sparse gradients')
                grads.append(p.grad)

                state = self.state[p]

                # State initialization
                if 'step' not in state:
                    state['step'] = torch.zeros((), dtype=torch.float, device=p.device)
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avgs.append(state['exp_avg'])
                exp_avg_sqs.append(state['exp_avg_sq'])
                if amsgrad:
                    max_exp_avg_sqs.append(state['max_exp_avg_sq'])

                # Update the steps for each param group update
                state['step'] += 1
                # Record the step after step update
                state_steps.append(state['step'])

            self.adamw(
                params_with_grad,
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
                eps=eps,
            )

        return loss

    def dist_reduce_metrics(self, optimizer_metrics):
        local_keys = list(optimizer_metrics.keys())
        all_gathered_keys = dist.all_gather_object(local_keys)
        all_keys = set()
        for keys in all_gathered_keys:
            all_keys.update(keys)

        # Sort keys to ensure every rank has the same keys order
        # Only L2 norm metric keys are present, can apply regular sort
        all_keys = sorted(all_keys)
        for metric in all_keys:
            if metric.startswith('l2_norm'):
                reduced = optimizer_metrics.get(metric, torch.tensor(0.0, device=torch.cuda.current_device()))
                if dist.get_world_size() > 1:
                    dist.all_reduce(reduced, reduce_operation='SUM')

                optimizer_metrics[metric] = math.sqrt(reduced)
            else:
                reduced = optimizer_metrics.get(metric, torch.tensor(0.0, device=torch.cuda.current_device()))
                if dist.get_world_size() > 1:
                    dist.all_reduce(reduced, reduce_operation='SUM')
                optimizer_metrics[metric] = reduced / dist.get_world_size()

        return optimizer_metrics

    def pre_reduce_metrics(self, optimizer_metrics):
        """Preprocess metrics to reduce across ranks correctly."""
        # Only L2 norm metric keys are present, can skip sorting at this stage
        for metric in optimizer_metrics:
            # L2 norms need to be squared, before they are reduced via summation
            optimizer_metrics[metric] = optimizer_metrics[metric]**2

        return optimizer_metrics

    def report_per_parameter_metrics(self, param: torch.Tensor, name: str, optimizer_metrics: dict):
        lr = self.param_groups[0]['lr']
        eps = self.param_groups[0]['eps']
        weight_decay = self.param_groups[0]['weight_decay']
        initial_lr = self.param_groups[0]['initial_lr']

        beta1, beta2 = self.param_groups[0]['betas']
        if param in self.state:
            param_optim_state = self.state[param]
            step = param_optim_state['step'].item()
            bias_correction1 = 1 - beta1**step
            bias_correction2 = 1 - beta2**step
            denom = (param_optim_state['exp_avg_sq'].sqrt() / math.sqrt(bias_correction2)).add_(eps)
            step_size = lr / bias_correction1
            step_tensor = step_size * param_optim_state['exp_avg'].div(denom)
            decay_factor = (lr / initial_lr) if initial_lr else 1.0
            step_tensor.add_(param, alpha=-weight_decay * decay_factor)
            for metric in self.metric_functions:
                optimizer_metrics[f'{metric}/{name}'] = self.metric_functions[metric](
                    param,
                    param_optim_state,
                    step_tensor,
                )

        return optimizer_metrics
