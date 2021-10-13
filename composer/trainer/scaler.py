# Copyright 2021 MosaicML. All Rights Reserved.

from collections import defaultdict
from typing import Callable

import torch
from torch.cuda.amp.grad_scaler import GradScaler, OptState, _refresh_per_optimizer_state
from torch.optim import Optimizer

from composer.core.types import Tensor


class ClosureGradScaler(GradScaler):

    def __init__(self, ddp_reduce_scalar_and: Callable[[bool], bool], ddp_reduce_tensor_sum: Callable[[Tensor], Tensor],
                 **kwargs):
        self.ddp_reduce_scalar_and = ddp_reduce_scalar_and
        self.ddp_reduce_tensor_sum = ddp_reduce_tensor_sum
        super().__init__(**kwargs)

    def _force_scaler_ready(self, optimizer: Optimizer):
        optimizer_state = self._per_optimizer_states[id(optimizer)]
        optimizer_state["stage"] = OptState.READY

    def _empty_all_grads(self, optimizer):
        for group in optimizer.param_groups:
            for param in group["params"]:
                if param.grad is not None:
                    param.grad = None

    def _unscale_grads_and_continue(self, optimizer: Optimizer):
        if (not self._enabled):
            return True
        self._check_scale_growth_tracker("step")
        optimizer_state = self._per_optimizer_states[id(optimizer)]

        if optimizer_state["stage"] is OptState.STEPPED:
            raise RuntimeError("step() has already been called since the last update().")

        if optimizer_state["stage"] is OptState.READY:
            self.unscale_(optimizer)
        inf_detected = sum(v.item() for v in optimizer_state["found_inf_per_device"].values())
        return not inf_detected

    def step(self, optimizer: Optimizer, *args, **kwargs):
        """ Always called before the optimizer step.
        Checks if the optimizer can handle AMP closures (currently only MosaicML's SAM optimizer)
        If so, it passes an AMP-modified closure to the optimizer.
        """

        closure = kwargs["closure"]

        def _amp_closure(**kwargs):
            self._force_scaler_ready(optimizer)
            self._empty_all_grads(optimizer)

            retval: float = closure(**kwargs)

            should_continue = self._unscale_grads_and_continue(optimizer)
            should_continue = self.ddp_reduce_scalar_and(should_continue)

            return retval if should_continue else None

        return optimizer.step(closure=_amp_closure)  # type: ignore

    # Mostly copied from original grad_scaler implementation
    def update(self, new_scale=None):
        """
        Updates the scale factor.

        If any optimizer steps were skipped the scale is multiplied by ``backoff_factor``
        to reduce it. If ``growth_interval`` unskipped iterations occurred consecutively,
        the scale is multiplied by ``growth_factor`` to increase it.

        Passing ``new_scale`` sets the new scale value manually. (``new_scale`` is not
        used directly, it's used to fill GradScaler's internal scale tensor. So if
        ``new_scale`` was a tensor, later in-place changes to that tensor will not further
        affect the scale GradScaler uses internally.)

        Args:
            new_scale (float or :class:`torch.cuda.FloatTensor`, optional, default=None):  New scale factor.

        .. warning::
            :meth:`update` should only be called at the end of the iteration, after ``scaler.step(optimizer)`` has
            been invoked for all optimizers used this iteration.
        """
        if not self._enabled:
            return

        _scale, _growth_tracker = self._check_scale_growth_tracker("update")

        if new_scale is not None:
            # Accept a new user-defined scale.
            if isinstance(new_scale, float):
                self._scale.fill_(new_scale)  # type: ignore[union-attr]
            else:
                reason = "new_scale should be a float or a 1-element torch.cuda.FloatTensor with requires_grad=False."
                assert isinstance(new_scale, torch.cuda.FloatTensor), reason  # type: ignore[attr-defined]
                assert new_scale.numel() == 1, reason
                assert new_scale.requires_grad is False, reason
                self._scale.copy_(new_scale)  # type: ignore[union-attr]
        else:
            # Consume shared inf/nan data collected from optimizers to update the scale.
            # If all found_inf tensors are on the same device as self._scale, this operation is asynchronous.
            found_infs = [
                found_inf.to(device=_scale.device, non_blocking=True)
                for state in self._per_optimizer_states.values()
                for found_inf in state["found_inf_per_device"].values()
            ]

            assert len(found_infs) > 0, "No inf checks were recorded prior to update."

            found_inf_combined = found_infs[0]
            if len(found_infs) > 1:
                for i in range(1, len(found_infs)):
                    found_inf_combined += found_infs[i]

            # This is the only line changed from original grad_scaler implementation
            found_inf_combined = self.ddp_reduce_tensor_sum(found_inf_combined)

            torch._amp_update_scale_(_scale, _growth_tracker, found_inf_combined, self._growth_factor,
                                     self._backoff_factor, self._growth_interval)

        # To prepare for next iteration, clear the data collected from optimizers this iteration.
        self._per_optimizer_states = defaultdict(_refresh_per_optimizer_state)
