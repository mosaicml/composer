# Copyright 2021 MosaicML. All Rights Reserved.

"""Core code for Stochastic Weight Averaging."""

from __future__ import annotations

import logging
from typing import Optional

import torch
from torch.optim.swa_utils import SWALR, AveragedModel

from composer.core.time import Time, TimeUnit
from composer.core.types import Algorithm, Event, Logger, State

log = logging.getLogger(__name__)

__all__ = ['SWA']


class SWA(Algorithm):
    """Apply Stochastic Weight Averaging (`Izmailov et al, 2018 <https://arxiv.org/abs/1803.05407>`_)

    Stochastic Weight Averaging (SWA) averages model weights sampled at
    different times near the end of training. This leads to better
    generalization than just using the final trained weights.

    Because this algorithm needs to maintain both the current value of the
    weights and the average of all of the sampled weights, it doubles the
    model's memory consumption. Note that this does not mean that the total
    memory required doubles, however, since stored activations and the
    optimizer state are not doubled.

    Uses PyTorch's `torch.optim.swa_util
    <https://pytorch.org/docs/stable/optim.html#stochastic-weight-averaging>`_ under the
    hood.

    See the :doc:`Method Card </method_cards/swa>` for more details.

    Example:
        .. testcode::

            from composer.algorithms import SWA
            from composer.trainer import Trainer

            swa_algorithm = SWA(
                swa_start="6ep",
                swa_end="8ep"
            )
            trainer = Trainer(
                model=model,
                train_dataloader=train_dataloader,
                eval_dataloader=eval_dataloader,
                max_duration="10ep",
                algorithms=[swa_algorithm],
                optimizers=[optimizer]
            )

    Args:
        swa_start (str, optional): The time string denoting the amount of training
            completed before stochastic weight averaging begins. Currently only units of
            duration ('dur') and epoch ('ep') are supported. Defalt = ``'0.7dur'``.
        swa_end (str, optional): The time string denoting the amount of training
            completed before the baseline (non-averaged) model is replaced with the
            stochastic weight averaged model. It's important to have at least one epoch
            of training after the baseline model is replaced by the SWA model so that the
            SWA model can have its buffers (most importantly its batch norm statistics)
            updated. If ``swa_end`` occurs during the final epoch of training (e.g.
            ``swa_end = 0.9dur`` and ``max_duration = "5ep"``, or ``swa_end = 1.0dur``),
            the SWA model will not have its buffers updated, which can negatively impact
            accuracy, so ensure ``swa_end`` < :math:`\\frac{N_{epochs}-1}{N_{epochs}}`.
            Currently only units of duration ('dur') and epoch ('ep') are supported.
            Default = ``'0.97dur'``.
        update_interval (str, optional): Time string denoting how often the averaged
            model is updated. For example, ``'1ep'`` means the averaged model will be
            updated once per epoch, and ``'5ba'`` means the averaged model will be updated
            every 5 batches. Note that for single-epoch training runs (e.g. many NLP
            training runs) ``update_interval`` must be specified in units of ``'ba'``,
            otherwise SWA won't happen. Also note that very small update intervals (e.g.
            ``"1ba"``) can substantially slow down training. Default = ``'1ep'``.
        schedule_swa_lr (bool, optional): Flag to determine whether to apply an
            SWA-specific LR schedule during the period in which SWA is active. Default =
            ``False``.
        anneal_strategy (str, optional): SWA learning rate annealing schedule strategy.
            "linear" for linear annealing, "cos" for cosine annealing. Default =
            ``"linear"``.
        anneal_steps (int, optional): Number of SWA model updates over which to
            anneal SWA learning rate. Note that updates are determined by the
            ``update_interval`` argument. For example, if ``anneal_steps = 10`` and
            ``update_interval = '1ep'``, then the SWA LR will be annealed once per epoch
            for 10 epochs; if ``anneal_steps = 20`` and ``update_interval = '8ba'``, then
            the SWA LR will be annealed once every 8 batches over the course of 160
            batches (20 steps * 8 batches/step). Default = ``10``.
        swa_lr (float, optional): The final learning rate to anneal towards with the SWA
            LR scheduler. Set to ``None`` for no annealing. Default = ``None``.
    """

    def __init__(self,
                 swa_start: str = "0.7dur",
                 swa_end: str = "0.97dur",
                 update_interval: str = "1ep",
                 schedule_swa_lr: bool = False,
                 anneal_strategy: str = "linear",
                 anneal_steps: int = 10,
                 swa_lr: Optional[float] = None):
        self.schedule_swa_lr = schedule_swa_lr
        self.anneal_strategy = anneal_strategy
        self.anneal_steps = anneal_steps
        self.swa_lr = swa_lr
        self.swa_model: Optional[torch.nn.Module] = None
        self.swa_completed = False

        # Check timestrings are parsable and convert into time objects
        try:
            self.swa_start = Time.from_timestring(swa_start)
        except ValueError as error:
            raise ValueError(f"Invalid time string for parameter swa_start") from error
        try:
            self.swa_end = Time.from_timestring(swa_end)
        except ValueError as error:
            raise ValueError(f"Invalid time string for parameter swa_end") from error
        try:
            self.update_interval = Time.from_timestring(update_interval)
        except ValueError as error:
            raise ValueError(f"Invalid time string for parameter update_interval") from error

        # Check time objects have supported units
        for time_attr in ["swa_start", "swa_end"]:
            time_obj = getattr(self, time_attr)
            if time_obj.unit not in [TimeUnit.DURATION, TimeUnit.EPOCH]:
                raise ValueError(f"Invalid unit string for parameter {time_attr}: {time_obj.unit}")
        if self.update_interval.unit not in [TimeUnit.BATCH, TimeUnit.EPOCH]:
            raise ValueError(f"Invalid unit string for parameter update_interval: "
                             f"{self.update_interval.unit}")

        # Check time objects have valid values
        if self.swa_start.unit == TimeUnit.DURATION:
            if self.swa_start < 0 or self.swa_start >= 1:
                raise ValueError("If swa_start is specified in units of 'dur', it must "
                                 "be in the interval [0, 1).")
        if self.swa_end.unit == TimeUnit.DURATION:
            if self.swa_end == 1:
                log.warning("'swa_end' = '1dur'. Batch norm statistics of averaged model "
                            "will not be updated. This will negatively impact accuracy. "
                            "See the documentation for the `swa_end` parameter for details.")
            if self.swa_end > 1:
                raise ValueError("If swa_end is specified in units of 'dur', it must be ≤1.")
        if self.update_interval < 1:
            raise ValueError("update_interval must be ≥ 1.")

        if anneal_steps <= 0:
            raise ValueError("anneal_steps must be greater than 0")

        # Check annealing_strategy string
        if self.anneal_strategy.lower() in ["linear", "lin"]:
            self.anneal_strategy = "linear"
        elif self.anneal_strategy.lower() in ["cos", "cosine"]:
            self.anneal_strategy = "cos"
        else:
            raise ValueError("Parameter 'anneal_strategy' must have an argument that is one of {'linear', 'cos'}.")

        self.swa_scheduler = None
        self.swa_model = None

        # Keeps track of # steps so that we can know when to update averaged model
        self.step_counter = None

        # Check units for update_interval and set match event accordingly
        if self.update_interval.unit == TimeUnit.BATCH:
            self.match_event = Event.BATCH_END
        elif self.update_interval.unit == TimeUnit.EPOCH:
            self.match_event = Event.EPOCH_END

    def match(self, event: Event, state: State) -> bool:
        if self.swa_start.unit == TimeUnit.DURATION:
            should_start_swa = state.get_elapsed_duration() >= self.swa_start and not self.swa_completed
        elif self.swa_start.unit == TimeUnit.EPOCH:
            should_start_swa = state.timer.get("ep") >= self.swa_start and not self.swa_completed
        else:
            should_start_swa = False
        return event == self.match_event and should_start_swa

    def apply(self, event: Event, state: State, logger: Logger) -> None:
        if self.step_counter is None:
            self.step_counter = 0

        if self.swa_scheduler is None and self.schedule_swa_lr:

            if self.swa_lr is None:
                if len(state.schedulers) != 1:
                    raise RuntimeError("SWA supports only one scheduler")
                scheduler = state.schedulers[0]
                last_lr = scheduler.get_last_lr()
                if len(last_lr) != 1:
                    raise RuntimeError(f"SWA supports only one LR; instead found {len(last_lr)}")
                log.info(f'Setting SWA LR to {last_lr}')
                self.swa_lr = last_lr[0]

            if len(state.optimizers) != 1:
                raise RuntimeError("SWA supports one and only one optimizer")

            self.swa_scheduler = SWALR(
                state.optimizers[0],
                swa_lr=self.swa_lr,
                anneal_epochs=self.anneal_steps,
                anneal_strategy=self.anneal_strategy,
            )

        if self.step_counter % self.update_interval.value == 0:
            if self.swa_model is None:
                self.swa_model = AveragedModel(state.model)

            self.swa_model.update_parameters(state.model)  # type: ignore

            if self.schedule_swa_lr:
                if self.swa_scheduler is None:
                    raise ValueError('SWA LR scheduler was not set.')
                self.swa_scheduler.step()

        self.step_counter += 1

        # Determine whether it's time to end SWA
        if self.swa_end.unit == TimeUnit.DURATION and (state.get_elapsed_duration() >= self.swa_end):
            self.swa_completed = True
        if self.swa_end.unit == TimeUnit.EPOCH and (state.timer.get("ep") >= self.swa_end):
            self.swa_completed = True
        if self.swa_completed:
            if state.get_elapsed_duration() == 1:
                log.warning("The baseline model was replaced with the SWA model after the end of "
                            "training. This means that SWA model will not have its batch norm "
                            "statistics updated. This will negatively impact accuracy. See the "
                            "documentation for the `swa_end` parameter for details.")
            state.model.load_state_dict(self.swa_model.module.state_dict())  # type: ignore
            log.info('Set model to the averaged model')
