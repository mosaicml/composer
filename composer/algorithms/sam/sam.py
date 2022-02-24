# Copyright 2021 MosaicML. All Rights Reserved.

from __future__ import annotations

import logging
from typing import Optional

import torch

from composer.core import Algorithm, Event, Logger, State
from composer.utils import ensure_tuple

log = logging.getLogger(__name__)


class SAMOptimizer(torch.optim.Optimizer):
    """Wraps an optimizer with sharpness-aware minimization (`Foret et al, 2020 <https://arxiv.org/abs/2010.01412>`_).
    See :class:`SAM` for details.

    Implementation based on https://github.com/davda54/sam
    """

    def __init__(self, base_optimizer, rho: float = 0.05, epsilon: float = 1.0e-12, interval: int = 1, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"
        self.base_optimizer = base_optimizer
        self.global_step = 0
        self.interval = interval
        self._step_supports_amp_closure = True  # Flag for Composer trainer
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

    @torch.no_grad()
    def first_step(self):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + group["epsilon"])
            for p in group["params"]:
                if p.grad is None:
                    continue
                e_w = p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"
                self.state[p]["e_w"] = e_w

    @torch.no_grad()
    def second_step(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None or "e_w" not in self.state[p]:
                    continue
                p.sub_(self.state[p]["e_w"])  # get back to "w" from "w + e(w)"
        self.base_optimizer.step()  # do the actual "sharpness-aware" update

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass
        loss = None

        if (self.global_step + 1) % self.interval == 0:
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
        rho: The neighborhood size parameter of SAM. Must be greater than 0.
        epsilon: A small value added to the gradient norm for numerical stability.
        interval: SAM will run once per ``interval`` steps. A value of 1 will
            cause SAM to run every step. Steps on which SAM runs take
            roughly twice as much time to complete.
    """

    def __init__(
        self,
        rho: float = 0.05,
        epsilon: float = 1.0e-12,
        interval: int = 1,
    ):
        """__init__ is constructed from the same fields as in hparams."""
        self.rho = rho
        self.epsilon = epsilon
        self.interval = interval

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

        state.optimizers = tuple(
            SAMOptimizer(
                base_optimizer=optimizer,
                rho=self.rho,
                epsilon=self.epsilon,
                interval=self.interval,
            ) for optimizer in ensure_tuple(state.optimizers))
