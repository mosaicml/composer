# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Core code for Stochastic Weight Averaging."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import torch
from torch.optim.swa_utils import SWALR, AveragedModel

from composer.core import Algorithm, Event, PyTorchScheduler, State, Time, TimeUnit
from composer.loggers import Logger

log = logging.getLogger(__name__)

__all__ = ['SWA']


def _assert_valid_duration(time: Time):
    if time.unit == TimeUnit.DURATION and (time < 0 or time > 1):
        raise ValueError(f'time in duration units must be [0, 1], got {time}')


class SWA(Algorithm):
    """Applies Stochastic Weight Averaging (`Izmailov et al, 2018 <https://arxiv.org/abs/1803.05407>`_).

    Stochastic Weight Averaging (SWA) averages model weights sampled at
    different times near the end of training. This leads to better
    generalization than just using the final trained weights.

    Because this algorithm needs to maintain both the current value of the
    weights and the average of all of the sampled weights, it doubles the
    model's memory consumption. Note that this does not mean that the total
    memory required doubles, however, since stored activations and the
    optimizer state are not doubled.

    .. note::

       The AveragedModel is currently stored on the CPU device, which may
       cause slow training if the model weights are large.

    Uses PyTorch's `torch.optim.swa_util
    <https://pytorch.org/docs/stable/optim.html#stochastic-weight-averaging>`_
    under the hood.

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
            duration ('dur') and epoch ('ep') are supported. Default: ``'0.7dur'``.
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
            Default: ``'0.97dur'``.
        update_interval (str, optional): Time string denoting how often the averaged
            model is updated. For example, ``'1ep'`` means the averaged model will be
            updated once per epoch and ``'5ba'`` means the averaged model will be updated
            every 5 batches. Note that for single-epoch training runs (e.g. many NLP
            training runs), ``update_interval`` must be specified in units of ``'ba'``,
            otherwise SWA won't happen. Also note that very small update intervals (e.g.
            ``"1ba"``) can substantially slow down training. Default: ``'1ep'``.
        schedule_swa_lr (bool, optional): Flag to determine whether to apply an
            SWA-specific LR schedule during the period in which SWA is active. Default:
            ``False``.
        anneal_strategy (str, optional): SWA learning rate annealing schedule strategy.
            ``"linear"`` for linear annealing, ``"cos"`` for cosine annealing. Default:
            ``"linear"``.
        anneal_steps (int, optional): Number of SWA model updates over which to
            anneal SWA learning rate. Note that updates are determined by the
            ``update_interval`` argument. For example, if ``anneal_steps = 10`` and
            ``update_interval = '1ep'``, then the SWA LR will be annealed once per epoch
            for 10 epochs; if ``anneal_steps = 20`` and ``update_interval = '8ba'``, then
            the SWA LR will be annealed once every 8 batches over the course of 160
            batches (20 steps * 8 batches/step). Default: ``10``.
        swa_lr (float, optional): The final learning rate to anneal towards with the SWA
            LR scheduler. Set to ``None`` for no annealing. Default: ``None``.
    """

    def __init__(self,
                 swa_start: str = '0.7dur',
                 swa_end: str = '0.97dur',
                 update_interval: str = '1ep',
                 schedule_swa_lr: bool = False,
                 anneal_strategy: str = 'linear',
                 anneal_steps: int = 10,
                 swa_lr: Optional[float] = None):
        self.schedule_swa_lr = schedule_swa_lr
        self.anneal_strategy = anneal_strategy
        self.anneal_steps = anneal_steps
        self.swa_lr = swa_lr
        self.swa_model: Optional[torch.nn.Module] = None
        self.swa_completed = False
        self.swa_started = False

        # Check timestrings are parsable and convert into time objects
        self.swa_start = Time.from_timestring(swa_start)
        self.swa_end = Time.from_timestring(swa_end)
        self.update_interval = Time.from_timestring(update_interval)

        self._validate_time()

        if anneal_steps <= 0:
            raise ValueError('anneal_steps must be greater than 0')

        # Check annealing_strategy string
        if self.anneal_strategy.lower() in ['linear', 'lin']:
            self.anneal_strategy = 'linear'
        elif self.anneal_strategy.lower() in ['cos', 'cosine']:
            self.anneal_strategy = 'cos'
        else:
            raise ValueError("anneal_strategy must be one of {'linear', 'cos'}.")

        self.swa_scheduler = None
        self.swa_model = None

        # Keeps track of # steps so that we can know when to update averaged model
        self.step_counter = 0

        # Check units for update_interval and set match event accordingly
        if self.update_interval.unit == TimeUnit.BATCH:
            self.match_event = Event.BATCH_END
        elif self.update_interval.unit == TimeUnit.EPOCH:
            self.match_event = Event.EPOCH_END

    def _validate_time(self):
        # validate time units
        if self.swa_start.unit != self.swa_end.unit:
            raise ValueError(f'swa_start and swa_end must have same units, got {self.swa_start} and {self.swa_end}')
        if self.swa_start.unit not in [TimeUnit.DURATION, TimeUnit.EPOCH]:
            raise ValueError(f'swa_start must be DURATION or EPOCH, got {self.swa_start.unit}')
        if self.update_interval.unit not in [TimeUnit.BATCH, TimeUnit.EPOCH]:
            raise ValueError(f'update_iterval must be BATCH or EPOCH, got {self.update_interval.unit}')

        # validate time
        if self.swa_start >= self.swa_end:
            raise ValueError('swa_end must be > swa_start.')
        if self.swa_end.unit == TimeUnit.DURATION and self.swa_end == 1:
            log.warning("'swa_end' = '1dur'. Batch norm statistics of averaged model "
                        'will not be updated. This will negatively impact accuracy. '
                        'See the documentation for the `swa_end` parameter for details.')

        _assert_valid_duration(self.swa_start)
        _assert_valid_duration(self.swa_end)

    def _get_time(self, state: State):
        """helper function to retrieve either the epoch or the duration depending on the units"""
        unit = self.swa_start.unit

        if unit == TimeUnit.EPOCH:
            return state.timestamp.epoch
        elif unit == TimeUnit.DURATION:
            time_elapsed = state.get_elapsed_duration()
            assert time_elapsed is not None, 'Time should have been set on BATCH_END or EPOCH_END.'
            return time_elapsed
        else:
            raise ValueError('units must be in epoch or duration.')

    def _get_last_lr(self, schedulers: List[PyTorchScheduler]):
        """ retrieves the last lr from current schedulers. """
        if len(schedulers) == 0:
            return 1.0
        if len(schedulers) != 1:
            raise RuntimeError(f'SWA supports only one scheduler, got {len(schedulers)}')
        scheduler = schedulers[0]
        last_lr = scheduler.get_last_lr()
        if len(last_lr) != 1:
            raise RuntimeError(f'SWA supports only one LR; instead found {len(last_lr)}')
        return last_lr[0]

    def match(self, event: Event, state: State) -> bool:
        if event == Event.INIT:
            return True

        # only match on BATCH_END or EPOCH_END, depending on the setting
        if event != self.match_event or self.swa_completed:
            return False

        return self._get_time(state) >= self.swa_start

    def _initialize_swa(self, state: State) -> None:
        if self.schedule_swa_lr:
            self.swa_lr = self._get_last_lr(state.schedulers)

            if len(state.optimizers) != 1:
                raise RuntimeError('SWA supports only one optimizer')

            self.swa_scheduler = SWALR(
                state.optimizers[0],
                swa_lr=self.swa_lr,
                anneal_epochs=self.anneal_steps,
                anneal_strategy=self.anneal_strategy,
            )

        self.swa_model = AveragedModel(state.model, device=torch.device('cpu'))

    def apply(self, event: Event, state: State, logger: Logger) -> None:

        if event == event.INIT:
            # on trainer init, we create the schedulers and models
            # so that the checkpoints can be loaded
            self._initialize_swa(state)
            return

        if not self.swa_started:
            # re-initialize swa once time > swa_start
            self._initialize_swa(state)
            self.swa_started = True

        if self.step_counter % self.update_interval.value == 0:
            assert self.swa_model is not None

            self.swa_model.update_parameters(state.model)  # type: ignore

            if self.schedule_swa_lr:
                assert self.swa_scheduler is not None
                self.swa_scheduler.step()

        self.step_counter += 1

        # Determine whether it's time to end SWA
        if self._get_time(state) >= self.swa_end:
            self.swa_completed = True

            if state.get_elapsed_duration() == 1:
                log.warning(('The baseline model was replaced with the SWA model after the end of '
                             'training. This means that SWA model will not have its batch norm '
                             'statistics updated. This will negatively impact accuracy. See the '
                             'documentation for the `swa_end` parameter for details.'))

            state.model.load_state_dict(self.swa_model.module.state_dict())  # type: ignore
            log.info('Set model to the averaged model')

    def state_dict(self) -> Dict[str, Any]:
        state_dict = super().state_dict()

        # we pop the anneal_func from the SWALR state
        # since it is set in the SWALR __init__
        swa_scheduler_state = None
        if self.swa_scheduler:
            swa_scheduler_state = self.swa_scheduler.state_dict()
            swa_scheduler_state.pop('anneal_func')

        state_dict = {
            'swa_model': self.swa_model.state_dict() if self.swa_model else None,
            'swa_completed': self.swa_completed,
            'swa_started': self.swa_started,
            'swa_scheduler': swa_scheduler_state,
            'step_counter': self.step_counter,
            **state_dict,
        }
        return state_dict

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.swa_completed = state['swa_completed']
        self.step_counter = state['step_counter']
        self.swa_started = state['swa_started']

        if self.swa_scheduler and state['swa_scheduler']:
            self.swa_scheduler.load_state_dict(state['swa_scheduler'])
        if self.swa_model and state['swa_model']:
            self.swa_model.load_state_dict(state['swa_model'])
