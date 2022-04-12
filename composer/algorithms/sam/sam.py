# Copyright 2021 MosaicML. All Rights Reserved.

from __future__ import annotations

import logging
from typing import Optional, Type, Union

import torch

from composer.core import Algorithm, Event, State
from composer.loggers import Logger
from composer.utils import ensure_tuple

from composer.algorithms.sam.sam_interval import (
    get_max_duration_as_steps,
    SAMInterval,
    SAM_FixedInterval,
    #SAM_Piecewise,
    #SAM_CosProb
)

log = logging.getLogger(__name__)


class SAMOptimizer(torch.optim.Optimizer):
    """Wraps an optimizer with sharpness-aware minimization (`Foret et al, 2020 <https://arxiv.org/abs/2010.01412>`_).
    See :class:`SAM` for details.

    Implementation based on https://github.com/davda54/sam

    Args:
        rho (float, optional): neighborhood size
        epsilon (float, optional): float to add to avoid division by 0
        interval (optional): either an int or an uninstantiated SAMInterval class.
            If an int, will set to using a fixed interval (SAM_FixedInterval).
            (Note: the reason it also supports an int is for backwards compatibility).
        num_max_steps (int, optional): the number of maximum steps taken during
            learning. Some SAMInterval classes require this. For ones that don't,
            (ex: SAM_FixedInterval) this parameter is ignored.
    """

    def __init__(self, base_optimizer, rho: float = 0.05,
                 epsilon: float = 1.0e-12,
                 #interval: int = 1,
                 interval: Union[int, Type[SAMInterval]] = 1,
                 num_max_steps:int = None, # only necessary for some interval schedules
                 use_LookSAM: bool = False,
                 alpha_LookSAM: float = 0.7,
                 use_ESAM_SWP: bool = False,
                 beta_ESAM_SWP: float = 0.6,
                 **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"
        self.base_optimizer = base_optimizer
        self.global_step = 0
        #self.interval = interval
        if isinstance(interval, int):
            self.interval_class = SAM_FixedInterval(num_max_steps, interval)
        else:
            self.interval_class = interval(num_max_steps, **kwargs)
        self._step_supports_amp_closure = True  # Flag for Composer trainer
        self.use_LookSAM = use_LookSAM # can also be added to param_groups if want to support doing LookSAM for only some layers
        self.alpha_LookSAM = alpha_LookSAM
        if self.use_LookSAM:
            self.gv_norm = None
        self.use_ESAM_SWP = use_ESAM_SWP # to support SDS from ESAM paper, need to either recompute loss or ensure that current loss reduction='none'
        self.beta_ESAM_SWP = beta_ESAM_SWP
        if self.use_ESAM_SWP:
            self.original_grad_states = {}
            # stores the original grad states to revert to

        defaults = dict(rho=rho, epsilon=epsilon, **kwargs)
        super(SAMOptimizer, self).__init__(self.base_optimizer.param_groups, defaults)

    @torch.no_grad()
    def sub_e_w(self):
        for group in self.param_groups:
            for p in group["params"]:
                if "e_w" not in self.state[p]:
                    continue
                e_w = self.state[p]["e_w"]  # retrieve stale e(w)
                p.sub_(e_w)  # get back to "w" from "w + e(w)"
                if self.use_LookSAM:
                    # don't use stored g_v
                    del self.state[p]['g_v']

    @torch.no_grad()
    def first_step(self):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + group["epsilon"])
            if self.use_ESAM_SWP:
                scale.div_(1-self.beta_ESAM_SWP)
            for p in group["params"]:
                if p.grad is None:
                    continue
                e_w = p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"
                self.state[p]["e_w"] = e_w
                if self.use_LookSAM:
                    # need to store original gradients for calculation of g_v
                    # this can be memory intensive and may require a better
                    # solution if RAM is of a concern
                    self.state[p]['prev_grad'] = p.grad.clone()

    @torch.no_grad()
    def second_step(self):
        if self.use_LookSAM:
            old_grad_norm = torch.norm(torch.stack(
                [self.state[p]['prev_grad'].norm(p=2) for group in self.param_groups for p in group["params"] if self.state[p]['prev_grad'] is not None]),
                            p="fro")
            new_grad_norm = self._grad_norm()
            grad_norm_prod = new_grad_norm * old_grad_norm
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None or "e_w" not in self.state[p]:
                    continue
                p.sub_(self.state[p]["e_w"])  # get back to "w" from "w + e(w)"
                if self.use_LookSAM:
                    cos_theta = (p.grad * self.state[p]['prev_grad']) / grad_norm_prod
                    self.state[p]['g_v'] = p.grad - new_grad_norm * cos_theta * self.state[p]['prev_grad'] / old_grad_norm

        if self.use_LookSAM:
            # calculate and store gv_norm once so don't need to recalculate
            # for future steps.
            self.gv_norm = torch.norm(torch.stack(
                [self.state[p]['g_v'].norm(p=2) for group in self.param_groups for p in group["params"] if self.state[p]['g_v'] is not None]),
                            p="fro")

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass
        loss = None
        #if (self.global_step + 1) % self.interval == 0:
        if self.interval_class.run_check(self.global_step):
            # Compute gradient at (w) per-GPU, and do not sync
            loss = closure(ddp_sync=False)  # type: ignore
            if loss:
                self.first_step()  # Compute e(w) and set weights to (w + (e(w)) separately per-GPU
                if closure():  # Compute gradient at (w + e(w))
                    self.second_step()  # Reset weights to (w) and step base optimizer
                else:
                    self.sub_e_w()  # If second forward-backward closure fails, reset weights to (w)
        else:
            loss = closure()
            if loss:
                if self.use_LookSAM:
                    grad_norm = self._grad_norm()
                    for group in self.param_groups:
                        for p in group["params"]:
                            if p.grad is None or 'g_v' not in self.state[p]:
                                continue
                            p.grad.add_(self.alpha_LookSAM * self.state[p]['g_v'] * grad_norm / self.gv_norm)
                self.base_optimizer.step()

        self.global_step += 1
        return loss

    def _grad_norm(self):
        norm = torch.norm(torch.stack(
            [p.grad.norm(p=2) for group in self.param_groups for p in group["params"] if p.grad is not None]),
                          p="fro")
        return norm


class SAM(Algorithm):
    """Adds sharpness-aware minimization (`Foret et al, 2020 <https://arxiv.org/abs/2010.01412>`_) by wrapping an
    existing optimizer with a :class:`SAMOptimizer`.

    Args:
        T (int):
        rho (float, optional): The neighborhood size parameter of SAM. Must be greater than 0.
            Default: ``0.05``.
        epsilon (float, optional): A small value added to the gradient norm for numerical stability.
            Default: ``1e-12``.
        interval (int, optional): SAM will run once per ``interval`` steps. A value of 1 will
            cause SAM to run every step. Steps on which SAM runs take
            roughly twice as much time to complete. Default: ``1``.
    """

    def __init__(
        self,
        rho: float = 0.05,
        epsilon: float = 1.0e-12,
        #interval: int = 1,
        interval: Union[int, Type[SAMInterval]] = 1,
        **kwargs
    ):
        """__init__ is constructed from the same fields as in hparams."""
        self.rho = rho
        self.epsilon = epsilon
        self.interval = interval
        self.kwargs = kwargs

    def match(self, event: Event, state: State) -> bool:
        """Run on Event.INIT.

        Args:
            event (:class:`Event`): The current event.
            state (:class:`State`): The current state.
        Returns:
            bool: True if this algorithm should run now
        """
        return event == Event.INIT

    def apply(self, event: Event, state: State, logger: Optional[Logger]) -> Optional[int]:
        """Applies SAM by wrapping the base optimizer with the SAM optimizer.

        Args:
            event (Event): the current event
            state (State): the current trainer state
            logger (Logger): the training logger
        """
        assert state.optimizers is not None

        num_max_steps = get_max_duration_as_steps(state)
        state.optimizers = tuple(
            SAMOptimizer(
                base_optimizer=optimizer,
                rho=self.rho,
                epsilon=self.epsilon,
                interval=self.interval,
                num_max_steps=num_max_steps,
                **self.kwargs
            ) for optimizer in ensure_tuple(state.optimizers))
