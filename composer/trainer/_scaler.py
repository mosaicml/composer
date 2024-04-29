# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

from collections import defaultdict
from typing import Optional, Union

import torch
from packaging import version
from torch.cuda.amp.grad_scaler import GradScaler, OptState
from torch.optim import Optimizer

if version.parse(torch.__version__) >= version.parse('2.3.0'):
    from torch.amp.grad_scaler import _refresh_per_optimizer_state  # type: ignore
else:
    from torch.cuda.amp.grad_scaler import _refresh_per_optimizer_state  # type: ignore

from composer.utils import dist

__all__ = ['ClosureGradScaler']


class ClosureGradScaler(GradScaler):
    """ClosureGradScaler allows for gradient scaling during with closures.

    We use closures with optimizers (see `here <https://pytorch.org/docs/stable/optim.html>`__)
    during training in order to support certain algorithms like
    :class:`~composer.algorithms.SAM`. This class allows us to perform gradient
    scaling (see `here <https://pytorch.org/docs/stable/amp.html#torch.cuda.amp.GradScaler>`__)
    along with the use of closures during training.

    Args:
        ddp_reduce_scalar_and (Callable[[bool], bool]): A function that performs a
            ddp reduction with an `and` operation. Used to determine whether
            or not to continue computing an optimizer's `step` based on the presence
            of `inf/nan` in the gradients.
        ddp_reduce_tensor_sum (Callable[[Tensor], Tensor]): A function that performs
            a ddp reduction across tensors with a `sum` operation. Used to aggregate
            `inf/nan` information stored in tensors across devices.
    """

    def _force_scaler_ready(self, optimizer: Optimizer):
        optimizer_state = self._per_optimizer_states[id(optimizer)]
        optimizer_state['stage'] = OptState.READY

    def _empty_all_grads(self, optimizer):
        for group in optimizer.param_groups:
            for param in group['params']:
                if param.grad is not None:
                    param.grad = None

    def _unscale_grads_and_continue(self, optimizer: Optimizer):
        if (not self._enabled):
            return True
        self._check_scale_growth_tracker('step')
        optimizer_state = self._per_optimizer_states[id(optimizer)]

        if optimizer_state['stage'] is OptState.STEPPED:
            raise RuntimeError('step() has already been called since the last update().')

        if optimizer_state['stage'] is OptState.READY:
            self.unscale_(optimizer)
        inf_detected = sum(v.item() for v in optimizer_state['found_inf_per_device'].values())
        return not inf_detected

    def step(self, optimizer: Optimizer, *args, **kwargs):
        """Step the optimizer with amp.

        Always called before the optimizer step. Checks if the optimizer can handle AMP closures (currently only
        Composer's SAM optimizer) If so, it passes an AMP-modified closure to the optimizer.
        """
        closure = kwargs['closure']

        def _amp_closure(**kwargs):
            self._force_scaler_ready(optimizer)
            self._empty_all_grads(optimizer)

            retval: float = closure(**kwargs)

            should_continue = self._unscale_grads_and_continue(optimizer)
            other_should_continue = dist.all_gather_object(should_continue)

            return retval if all(other_should_continue) else None

        return optimizer.step(closure=_amp_closure)  # type: ignore

    # Mostly copied from original grad_scaler implementation
    # See: https://pytorch.org/docs/stable/_modules/torch/amp/grad_scaler.html#GradScaler
    def update(self, new_scale: Optional[Union[float, torch.FloatTensor]] = None):
        """Updates the scale factor.

        If any optimizer steps were skipped, the scale is multiplied by ``backoff_factor``
        to reduce it. If ``growth_interval`` non-skipped iterations occurred consecutively,
        the scale is multiplied by ``growth_factor`` to increase it.

        Passing ``new_scale`` sets the new scale value manually. (``new_scale`` is not
        used directly; it is used to fill GradScaler's internal scale tensor. So, if
        ``new_scale`` was a tensor, later in-place changes to that tensor will not further
        affect the scale that the GradScaler uses internally.)

        .. warning::

            This method should only be called at the end of the iteration, after ``scaler.step(optimizer)`` has
            been invoked for all optimizers used this iteration.

        Args:
            new_scale (float | FloatTensor, optional):  New scale factor. (default: ``None``)
        """
        if not self._enabled:
            return

        _scale, _growth_tracker = self._check_scale_growth_tracker('update')

        if new_scale is not None:
            # Accept a new user-defined scale.
            if isinstance(new_scale, float):
                self._scale.fill_(new_scale)  # type: ignore[union-attr]
            else:
                reason = 'new_scale should be a float or a 1-element torch.cuda.FloatTensor with requires_grad=False.'
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
                for found_inf in state['found_inf_per_device'].values()
            ]

            assert len(found_infs) > 0, 'No inf checks were recorded prior to update.'

            found_inf_combined = found_infs[0]
            if len(found_infs) > 1:
                for i in range(1, len(found_infs)):
                    found_inf_combined += found_infs[i]

            # This is the only line changed from original grad_scaler implementation
            dist.all_reduce(found_inf_combined, reduce_operation='SUM')

            torch._amp_update_scale_(
                _scale,
                _growth_tracker,
                found_inf_combined,
                self._growth_factor,
                self._backoff_factor,
                self._growth_interval,
            )

        # To prepare for next iteration, clear the data collected from optimizers this iteration.
        self._per_optimizer_states = defaultdict(_refresh_per_optimizer_state)
