# Copyright 2021 MosaicML. All Rights Reserved.

import logging
from abc import ABC
from dataclasses import asdict, dataclass, fields
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch
import yahp as hp
from torch.optim.lr_scheduler import (CosineAnnealingLR, CosineAnnealingWarmRestarts, ExponentialLR, MultiStepLR,
                                      StepLR, _LRScheduler)

from composer.core.time import TimeUnit
from composer.core.types import Optimizer, Scheduler, Time
from composer.optim.pytorch_future import LinearLR, WarmUpLR
from composer.utils._time_conversion import convert as convert_time

log = logging.getLogger(__name__)

_interval_doc = 'frequency of step() calls, either "batch" or "epoch". Default: "epoch"'

# Allow (batch, batches) or (epoch, epochs). Also accept "step" ~ "batch"
INTERVAL_MAP = {
    'batch': 'batch',
    'batches': 'batch',
    'epoch': 'epoch',
    'epochs': 'epoch',
    'step': 'batch',
    'steps': 'batch'
}


def _parse_time_string(timestring: str) -> Tuple[float, int, int]:
    """Parse timestring to (duration, epoch, batches).

    Args:
        timestring (str): String in the format 'XXdurYYepZZba'.

    Returns:
        tuple: (duration, epochs, batches)

    Raises:
        ValueError: The timestring is invalid

    Examples:
        >>> _parse_time_string('0.98dur32ep173ba')
        (0.98, 32, 173)
        >>> _parse_time_string('32ep173ba')
        (0, 32, 173)
        >>> _parse_time_string('12ep')
        (0, 12, 0)
        >>> _parse_time_string('1024ba')
        (0, 0, 1024)
    """

    match = STR_REGEX.findall(timestring)
    if len(match) != 1:
        raise ValueError(f'Invalid timestring: {timestring}. Should be of format 0.98dur32ep15ba, or subsets thereof.')
    match = match[0]

    duration = 0 if 'dur' not in match else float(match[match.index('dur') - 1])
    epochs = 0 if 'ep' not in match else int(match[match.index('ep') - 1])
    batches = 0 if 'ba' not in match else int(match[match.index('ba') - 1])

    return duration, epochs, batches


def _convert_time(time: Time,
                  steps_per_epoch: Optional[int] = None,
                  max_epochs: Optional[int] = None,
                  interval: str = 'epoch') -> int:
    """Convert time to either batches or epochs (based on interval argument)."""
    if isinstance(time, int):
        return time
    if steps_per_epoch is None:
        raise ValueError('steps_per_epoch must be provided to parse time string.')

    duration, epochs, batches = _parse_time_string(time)

    if interval in ('batches', 'batch', 'steps', 'step'):
        new_time = batches + epochs * steps_per_epoch

        if duration > 0:
            assert max_epochs is not None
            total_duration = max_epochs * steps_per_epoch
            new_time += (total_duration * duration)

        new_time = int(round(new_time))
        print(f'Converting {time}, {interval} to {new_time}')
        return new_time
    elif interval in ('epochs', 'epoch'):
        if duration > 0:
            assert max_epochs is not None
            # convert the duration term into batches for ease of calculation
            # round batches to the nearest term
            batches += int(round(steps_per_epoch * max_epochs * duration))
        epochs = epochs + batches // steps_per_epoch
        batches = batches % steps_per_epoch
        if batches != 0:
            log.warning('Scheduler is stepping every epoch, but provided timestring '
                        f'{time} had extra batches. Ignoring the extra batches.')
        log.info(f'Converting {time}, {interval} to {epochs}')
        return epochs
    else:
        raise ValueError('interval must be one of (batch, epoch)')


@dataclass
class SchedulerHparams(hp.Hparams, ABC):

    scheduler_object = None  # type: Optional[Callable[..., Scheduler]]
    interval = 'epochs'  # type: str

    def _convert_time_fields(self,
                             max_training_duration: Optional[Union[str, Time[int]]] = None,
                             steps_per_epoch: Optional[int] = None,
                             samples_per_epoch: Optional[int] = None,
                             dataset_num_tokens: Optional[int] = None) -> None:
        """Converts all fields that were provided as timestrings (e.g. "32ep") into
        integers, representing either epochs or batches, depending on the
        :attr:`interval`. Updates the fields in-place."""
        interval_unit = TimeUnit(INTERVAL_MAP[self.interval])

        for field in fields(self):
            field_value = getattr(self, field.name)

            if field.name not in ('interval', 'warmup_method') and isinstance(field_value, str):
                time = getattr(self, field.name)
                if isinstance(time, list):
                    result = [
                        convert_time(t,
                                     unit=interval_unit,
                                     steps_per_epoch=steps_per_epoch,
                                     max_training_duration=max_training_duration,
                                     samples_per_epoch=samples_per_epoch,
                                     dataset_num_tokens=dataset_num_tokens).value for t in time
                    ]
                else:
                    result = convert_time(time,
                                          unit=interval_unit,
                                          steps_per_epoch=steps_per_epoch,
                                          max_training_duration=max_training_duration,
                                          samples_per_epoch=samples_per_epoch,
                                          dataset_num_tokens=dataset_num_tokens).value

                setattr(self, field.name, result)

    def initialize_object(
        self,
        optimizer: Optimizer,
        steps_per_epoch: Optional[int] = None,
        samples_per_epoch: Optional[int] = None,
        dataset_num_tokens: Optional[int] = None,
        max_training_duration: Optional[Union[str, Time[int]]] = None,
    ) -> Tuple[Scheduler, str]:
        """Create the scheduler object from the current hparams.

        Args:
            optimizer (Optimizer): the optimizer associated with this scheduler
            steps_per_epoch (int, optional): The number of optimization steps per epoch.
            samples_per_epoch (int, optional): The number of samples trained per epoch.
            dataset_num_tokens (int, optional): The number of tokens in the dataset.
            max_training_duration (str or Time, optional): The total training duration.
        Returns:
            (Scheduler, str): (The parametrized scheduler instance, schedule step interval)
        """

        assert self.scheduler_object is not None, "Scheduler Hparams needs scheduler_object to initialize."

        self._convert_time_fields(max_training_duration=max_training_duration,
                                  steps_per_epoch=steps_per_epoch,
                                  samples_per_epoch=samples_per_epoch,
                                  dataset_num_tokens=dataset_num_tokens)

        # we pass the interval to the trainer directly
        kwargs = {k: v for k, v in asdict(self).items() if k not in ['interval']}
        obj = self.scheduler_object(optimizer, **kwargs)
        obj.interval = self.interval  # type: ignore
        obj.steps_per_epoch = steps_per_epoch  # type: ignore
        return obj, self.interval


class ConstantLR(_LRScheduler):
    """Scheduler that does not change the optimizer's learning rate.

    Args:
        optimizer (Optimizer): the optimizer associated with this scheduler.
        last_epoch (int, optional): The index of the last epoch. Can be used to restore the state of the
                                    learning rate schedule. Default: ``-1``.
        verbose (bool, optional): If ``True``, prints a message to stdout for each update. Default: ``False``.
    """

    def __init__(self, optimizer: Optimizer, last_epoch: int = -1, verbose: int = False):

        self.optimizer = optimizer
        super(ConstantLR, self).__init__(optimizer, last_epoch, verbose)  # type: ignore

    def get_lr(self):
        """ Get the current learning rate for each parameter group.

        Returns:
            List of float: The current learning rate for each parameter group.
        """
        return self.base_lrs  # type: ignore

    def _get_closed_form_lr(self):
        """ Get the current learning rate for each parameter group.

        Returns:
            List of float: The current learning rate for each parameter group.
        """
        return [base_lr for base_lr in self.base_lrs]  # type: ignore


class PolynomialLR(_LRScheduler):
    """PolynomialLR scales the learning rate by the remaining train time percentage raised to a specific power.

    Args:
        optimizer (Optimizer): the optimizer associated with this scheduler.
        T_max (Time): the number of iterations to perform, either in terms of epochs or batches.
        power (float): the power to use on the remaining train time percentage for the current schedule coeffecient.
        eta_min (float): the minimum learning rate to decay to. Default is ``0``.
        last_epoch (int): the index of the last epoch. Can be used to restore the learning rate schedule state.
            Default: ``-1``
        verbose (bool): If ``True``, prints a message to stdout for each update. Default: ``False``.
    """

    def __init__(self,
                 optimizer: Optimizer,
                 T_max: Time,
                 power: float,
                 eta_min: float = 0,
                 last_epoch: int = -1,
                 verbose: bool = False):

        self.optimizer = optimizer
        self.T_max = T_max
        self.power = power
        self.eta_min = eta_min
        super(PolynomialLR, self).__init__(optimizer, last_epoch, verbose)  # type: ignore

    def get_lr(self):
        coeff = (1 - self.last_epoch / self.T_max)**self.power  # type: ignore
        return [(base_lr - self.eta_min) * coeff + self.eta_min for base_lr in self.base_lrs]  # type: ignore


@dataclass
class PolynomialLRHparams(SchedulerHparams):
    """Hyperparameters for the :class:`PolynomialLR` scheduler."""
    T_max: str = hp.required(doc='Maximum number of iterations.')
    power: float = hp.required(doc='Power of LR schedule.')
    eta_min: float = hp.optional(default=0.0, doc='Minimum learning rate.')
    verbose: bool = hp.optional(default=False, doc='Prints message to stdout.')
    interval: str = hp.optional(default='epoch', doc=_interval_doc)

    scheduler_object = PolynomialLR


@dataclass
class ConstantLRHparams(SchedulerHparams):
    """Hyperparameters for the :class:`ConstantLR` scheduler."""
    verbose: bool = hp.optional(default=False, doc='prints message to stdout')
    interval: str = hp.optional(default='epoch', doc=_interval_doc)

    scheduler_object = ConstantLR


@dataclass
class StepLRHparams(SchedulerHparams):
    """Hyperparameters for the `StepLR <https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.StepLR.html#torch.optim.lr_scheduler.StepLR>`_
    scheduler.
    """

    step_size: str = hp.required(doc='Period of learning rate decay')
    gamma: float = hp.optional(default=0.1, doc='multiplicative factor of decay')
    verbose: bool = hp.optional(default=False, doc='prints message to stdout')
    interval: str = hp.optional(default='epoch', doc=_interval_doc)

    scheduler_object = torch.optim.lr_scheduler.StepLR


@dataclass
class MultiStepLRHparams(SchedulerHparams):
    """Hyperparameters for the `MultiStepLR <https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.MultiStepLR.html#torch.optim.lr_scheduler.MultiStepLR>`_
    scheduler.
    """

    milestones: List[str] = hp.required(doc='List of milestone time strings')
    gamma: float = hp.optional(default=0.1, doc='multiplicative factor of decay')
    verbose: bool = hp.optional(default=False, doc='prints message to stdout')
    interval: str = hp.optional(default='epoch', doc=_interval_doc)

    scheduler_object = torch.optim.lr_scheduler.MultiStepLR


@dataclass
class ExponentialLRHparams(SchedulerHparams):
    """Hyperparameters for the `ExponentialLR <https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ExponentialLR.html#torch.optim.lr_scheduler.ExponentialLR>`_
    scheduler.
    """

    gamma: float = hp.required(doc='multiplicative factor of decay')
    verbose: bool = hp.optional(default=False, doc='prints message to stdout')
    interval: str = hp.optional(default='epoch', doc=_interval_doc)

    scheduler_object = torch.optim.lr_scheduler.ExponentialLR


@dataclass
class CosineAnnealingLRHparams(SchedulerHparams):
    """Hyperparameters for the `CosineAnnealingLR <https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html#torch.optim.lr_scheduler.CosineAnnealingLR>`_
    scheduler.
    """

    T_max: str = hp.required(doc="Maximum scheduler duration.")
    eta_min: float = hp.optional(default=0.0, doc='minimum learning rate.')
    verbose: bool = hp.optional(default=False, doc='prints message to stdout')
    interval: str = hp.optional(default='epoch', doc=_interval_doc)

    scheduler_object = torch.optim.lr_scheduler.CosineAnnealingLR


@dataclass
class CosineAnnealingWarmRestartsHparams(SchedulerHparams):
    """Hyperparameters for the ``CosineAnnealingWarmRestarts` <https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingWarmRestarts.html#torch.optim.lr_scheduler.CosineAnnealingWarmRestarts>`_
    scheduler.
    """

    T_0: str = hp.required("Duration for the first restart.")
    eta_min: float = hp.optional(default=0.0, doc='minimum learning rate.')
    verbose: bool = hp.optional(default=False, doc='prints message to stdout')
    interval: str = hp.optional(default='epoch', doc=_interval_doc)
    T_mult: int = hp.optional("A factor increases :math:`T_{i}` after a restart. Default: 1.", default=1)

    scheduler_object = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts


@dataclass
class LinearLRHparams(SchedulerHparams):
    """Hyperparameters for the `LinearLRHparams <https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.LinearLR.html>`_
    scheduler.
    """

    start_factor: float = hp.optional("Number to multiply learning rate at the start.", default=1.0 / 3)
    end_factor: float = hp.optional("Number to multiply learning rate at the end .", default=1.0)
    total_iters: str = hp.optional("Duration of linear decay steps. Default: 5 iterations.", default="5ba")
    verbose: bool = hp.optional('Prints message to stdout', default=False)
    interval: str = hp.optional(default='epoch', doc=_interval_doc)

    scheduler_object = LinearLR


@dataclass
class WarmUpLRHparams(SchedulerHparams):
    """Hyperparameters for the :class:`~composer.optim.pytorch_future.WarmUpLR` scheduler.

    See the documentation for :class:`~composer.optim.pytorch_future.WarmUpLR`.
    """

    warmup_factor: float = hp.optional("Number to multiply learning rate at start.", default=1.0 / 3)
    warmup_iters: str = hp.optional("Warmup duration. Default: 5 iterations.", default="5ba")
    warmup_method: str = hp.optional("Warmup method (linear or constant)", default='linear')
    verbose: bool = hp.optional('Prints message to stdout', default=False)
    interval: str = hp.optional('Warmup the LR every step or epoch. Default: epoch', default='epoch')

    scheduler_object = WarmUpLR


def ensure_warmup_last(schedulers: List[SchedulerHparams]) -> List[SchedulerHparams]:
    """Ensure that WarmUp-based schedulers appear last in the provided list.

    Args:
        schedulers (List[SchedulerHparams]): List of schedulers.

    Returns:
        List[SchedulerHparams]: A sorted list of schedulers with WarmUp-based schedulers at the end.
    """

    return sorted(schedulers, key=lambda x: isinstance(x, (WarmUpLR, WarmUpLRHparams)))


def get_num_warmup_batches(scheduler_hparams: Sequence[SchedulerHparams], steps_per_epoch: Optional[int] = None) -> int:
    """Gets the number of warmup steps declared by a list of schedulers.

    Args:
        scheduler_hparams (Sequence[SchedulerHparams]): List of schedulers
        steps_per_epoch (Optional[int], optional): Number of steps in a single epoch. Default: ``None``.

    Returns:
        int: Number of warmup steps
    """

    warmup_scheduler_hparams = [scheduler for scheduler in scheduler_hparams if isinstance(scheduler, WarmUpLRHparams)]
    if len(warmup_scheduler_hparams):
        warmup_iters = warmup_scheduler_hparams[0].warmup_iters
        if isinstance(warmup_iters, str):
            interval_unit = TimeUnit(INTERVAL_MAP[warmup_scheduler_hparams[0].interval])
            return convert_time(
                time=warmup_iters,
                unit=interval_unit,
                steps_per_epoch=steps_per_epoch,
            ).value
        else:
            return warmup_iters
    return 0


class ComposedScheduler(_LRScheduler):
    """Handles warmup for a chained list of schedulers.

    With one call, will run each scheduler's ``step()``. If :class:`WarmUpLR` is in the list, will delay the stepping of
    schedulers that need to be silent during warmup. ``ComposedScheduler`` handles warmups, where as `ChainedScheduler <https://pytorch.org/docs/1.10./generated/torch.optim.lr_scheduler.ChainedScheduler.html?highlight=chained#torch.optim.lr_scheduler.ChainedScheduler>`_
    only combines schedulers.

    `CosineAnnealingLR <https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html#torch.optim.lr_scheduler.CosineAnnealingLR>`_
    and `ExponentialLR <https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ExponentialLR.html#torch.optim.lr_scheduler.ExponentialLR>`_
    are not stepped during the warmup period. Other schedulers, such as
    `MultiStepLR <https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.MultiStepLR.html#torch.optim.lr_scheduler.MultiStepLR>`_
    are still stepped, to keep their milestones unchanged.

    Handles running the :class:`WarmUpLR` at every step if :attr:`WarmUpLR.interval='batch'`, and other schedulers at
    every epoch.

    Args:
        schedulers (list): List of chained schedulers.
    Example:
        >>> # Assuming optimizer uses lr = 1. for all groups
        >>> # lr = 0.1      if epoch == 0
        >>> # lr = 0.1      if epoch == 1
        >>> # lr = 0.9      if epoch == 2  # ExponentialLR effect starts here
        >>> # lr = 0.81     if epoch == 3
        >>> # lr = 0.729    if epoch == 4
        >>> scheduler1 = WarmUpLR(self.opt, warmup_factor=0.1, warmup_iters=2, warmup_method="constant")
        >>> scheduler2 = ExponentialLR(self.opt, gamma=0.9)
        >>> scheduler = ComposedScheduler(zip([scheduler1, scheduler2], ["epoch", "epoch"]))
        >>> for epoch in range(100):
        >>>     train(...)
        >>>     validate(...)
        >>>     scheduler.step()

        >>> # Assuming optimizer uses lr = 1. for all groups
        >>> # lr = 0.1      if epoch == 0
        >>> # lr = 0.1      if epoch == 1
        >>> # lr = 1.0      if epoch == 2
        >>> # lr = 1.0     if epoch == 3
        >>> # lr = 0.2    if epoch == 4 . # MultiStepLR effect starts here
        >>> scheduler1 = WarmUpLR(self.opt, warmup_factor=0.1, warmup_iters=2, warmup_method="constant")
        >>> scheduler2 = MultiStepLR(optimizer, milestones=[4], gamma=0.2)
        >>> scheduler = ComposedScheduler(zip([scheduler1, scheduler2], ["epoch", "epoch"]))
        >>> for epoch in range(100):
        >>>     train(...)
        >>>     validate(...)
        >>>     scheduler.step()
    """

    def __init__(self, schedulers):

        # check for tuple
        if not all(isinstance(scheduler, tuple) for scheduler in schedulers):
            raise ValueError('Schedulers must be a tuple of (Scheduler, interval), '
                             'where interval is one of "epoch" or "batch".')

        self._validate_same_optimizers(schedulers)
        self.schedulers, self.intervals = list(zip(*schedulers))  # unpack (scheduler, interval)

        # generous with spelling (batch, batches)/(step, steps) and (epoch, epochs)
        self.intervals = [INTERVAL_MAP[interval] for interval in self.intervals]

        warmup = [(scheduler, interval)
                  for scheduler, interval in zip(self.schedulers, self.intervals)
                  if isinstance(scheduler, WarmUpLR)]
        if warmup:
            assert len(warmup) == 1, "ComposedScheduler only supports one WarmUpLR " \
                                     f"in the provided list, found {len(warmup)}."
            warmup, interval = warmup[0]
            self.warmup_iters = warmup.warmup_iters
            log.info(f'Setting LR Warmup to {self.warmup_iters} {interval}')
        else:
            self.warmup_iters = 0

        # these schedulers need to be silent during warmup
        self.delay_schedulers = [CosineAnnealingLR, CosineAnnealingWarmRestarts, ExponentialLR, LinearLR]
        self._warmup_counter = 0  # counter to track warmups

    def step(self, interval: str = 'epoch'):
        """Step all applicable schedulers.

        Args:
            interval (str, optional): The interval of the current step. Must be either ``'step'`` or ``'epoch'``.
                                      Default: ``epoch``.
        """
        for scheduler, scheduler_interval in zip(self.schedulers, self.intervals):
            if self._warmup_counter < self.warmup_iters and \
                any(isinstance(scheduler, delay) for delay in self.delay_schedulers):
                continue

            if interval == scheduler_interval:
                scheduler.step()
                if isinstance(scheduler, WarmUpLR):
                    self._warmup_counter += 1

    def _validate_schedulers(self, warmup_epochs: int) -> None:
        """Verify that any stepwise schedulers do not change the LR during the desired warmup period.

        Args:
            warmup_epochs (int): Number of epochs for warmup.
        """
        # since WarmUpLR is non-chainable form, step LR milestones must
        # occur after warmup is completed
        lr_step_schedulers = [
            scheduler for scheduler in self.schedulers if isinstance(scheduler, (StepLR, MultiStepLR))
        ]
        for scheduler in lr_step_schedulers:
            if isinstance(scheduler, StepLR) and scheduler.step_size <= warmup_epochs:  # type: ignore
                raise ValueError(f'StepLR step_size {scheduler.step_size} must '  # type: ignore
                                 'be greater than warmup_iters {self.warmup_iters}')
            elif isinstance(scheduler, MultiStepLR):
                if any(ms <= warmup_epochs for ms in scheduler.milestones.elements()):  #type: ignore
                    raise ValueError(f'MultiStepLR milestones must be greater than warmup_iters {warmup_epochs}')

    def state_dict(self) -> Dict[str, Any]:
        """Returns a dictionary containing the state of all composed schedulers.

        Returns:
            Dict: the state dictionary
        """
        state_dict = {
            "schedulers": {scheduler.__class__.__qualname__: scheduler.state_dict() for scheduler in self.schedulers},
            "_warmup_counter": self._warmup_counter,
        }
        return state_dict

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load the state of all composed schedulers from the provided dictionary.

        Args:
            state_dict (Dict[str, Any]): A dict containing the state of all composed schedulers. Should be an object
            returned from a call to :meth:`state_dict()`.
        """
        for scheduler in self.schedulers:
            scheduler.load_state_dict(state_dict["schedulers"][scheduler.__class__.__qualname__])
        self._warmup_counter = state_dict["_warmup_counter"]

    def _validate_same_optimizers(self, schedulers):
        """Verify that all schedulers correspond to the same optimizer."""
        for scheduler_idx in range(1, len(schedulers)):
            if (schedulers[scheduler_idx][0].optimizer != schedulers[0][0].optimizer):  # type: ignore
                raise ValueError("ComposedScheduler expects all schedulers to belong to the same optimizer, but "
                                 "got schedulers at index {} and {} to be different".format(0, scheduler_idx))
