from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Optional

import torch
import yahp as hp

from composer.algorithms import AlgorithmHparams
from composer.core import Algorithm, Event, Logger, State
from composer.utils import ensure_tuple

log = logging.getLogger(__name__)


@dataclass
class SAMHparams(AlgorithmHparams):
    rho: float = hp.optional(doc='The neighborhood size parameter of SAM. Must be greater than 0.', default=0.05)
    epsilon: float = hp.optional(doc='A small value added to denominators for numerical stability.', default=1.0e-12)
    adaptive: bool = hp.optional(doc='Whether to use the adaptive variant of SAM.', default=False)
    interval: int = hp.optional(doc='SAM will run once per `interval` steps. A value of 1 will cause'
                                'SAM to run every step. Steps on which SAM runs take roughly twice'
                                'as much time to complete.',
                                default=1)
    use_stale: bool = hp.optional(doc='If true, on steps where SAM would not normally run, use stale'
                                  'epsilon values from the previous invocation to run the algorithm'
                                  'without incurring a substantial throughput penalty.',
                                  default=False)

    def initialize_object(self) -> SAM:
        return SAM(**asdict(self))


class SAMOptimizer(torch.optim.Optimizer):
    """Implementation based on https://github.com/davda54/sam"""

    def __init__(self, base_optimizer, rho, epsilon, adaptive, interval, use_stale, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"
        self.base_optimizer = base_optimizer
        self.global_step = 0
        self.interval = interval
        self.use_stale = use_stale
        self._step_supports_amp_closure = True  # Flag for Mosaic trainer
        defaults = dict(rho=rho, epsilon=epsilon, adaptive=adaptive, **kwargs)
        super(SAMOptimizer, self).__init__(self.base_optimizer.param_groups, defaults)

    @torch.no_grad()
    def add_e_w(self):
        for group in self.param_groups:
            for p in group["params"]:
                if "e_w" not in self.state[p]:
                    continue
                e_w = self.state[p]["e_w"]  # retrieve stale e(w)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"

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
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
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
            if closure(ddp_sync=False):  # Use separate e(w) per-GPU
                self.first_step()
                loss = closure()
                if loss:
                    self.second_step()
                else:
                    self.sub_e_w()
        else:
            if self.use_stale:
                self.add_e_w()
                loss = closure()
                if loss:
                    self.second_step()
                else:
                    self.sub_e_w()
            else:
                loss = closure()
                if loss:
                    self.base_optimizer.step()

        self.global_step += 1
        return loss

    def _grad_norm(self):
        norm = torch.norm(torch.stack([((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2)
                                       for group in self.param_groups
                                       for p in group["params"]
                                       if p.grad is not None]),
                          p=2)
        return norm


class SAM(Algorithm):
    """Applies SAM by wrapping existing optimizers with the SAMOptimizer."""

    def __init__(self,
                 rho: float = 0.05,
                 epsilon: float = 1.0e-12,
                 adaptive: bool = False,
                 interval: int = 1,
                 use_stale: bool = False):
        """
        __init__ is constructed from the same fields as in hparams.
        """
        self.hparams = SAMHparams(rho=rho, epsilon=epsilon, adaptive=adaptive, interval=interval, use_stale=use_stale)

    def match(self, event: Event, state: State) -> bool:
        return event == Event.TRAINING_START

    def apply(self, event: Event, state: State, logger: Optional[Logger]) -> Optional[int]:
        assert state.optimizers is not None

        state.optimizers = tuple(
            SAMOptimizer(
                base_optimizer=optimizer,
                rho=self.hparams.rho,
                epsilon=self.hparams.epsilon,
                adaptive=self.hparams.adaptive,
                interval=self.hparams.interval,
                use_stale=self.hparams.use_stale,
            ) for optimizer in ensure_tuple(state.optimizers))
