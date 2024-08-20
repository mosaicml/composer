# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Utilities to track training progress in terms of epochs, batches, samples, and tokens.

Callbacks, algorithms, and schedulers can use the current training time to fire at certain points in the training
process.

The :class:`~.time.Timestamp` class tracks the total number of epochs, batches, samples, and tokens. The trainer is
responsible for updating it at the end of every epoch and batch.  There is only one instance of the
:class:`~.time.Timestamp`, which is attached to the :class:`~.state.State`.

The :class:`~.time.Time` class represents static durations of training time or points in the training process in terms
of a specific :class:`~.time.TimeUnit` enum. This class supports comparisons, arithmetic, and conversions.

See the :doc:`Time Guide </trainer/time>` for more details on tracking time during training.
"""
from __future__ import annotations

import datetime
import re
from typing import Any, Generic, Optional, TypeVar, Union, cast

from composer.core.serializable import Serializable
from composer.utils import StringEnum

__all__ = ['TimeUnit', 'Time', 'Timestamp', 'ensure_time']


def verify_wct(timestamp: str) -> str:
    """Return a valid datetime formated wct timestamp if input is a valid wct.

    Args:
        timestamp (str): A string that represents a timestamp in wct.

    Returns:
        str: a properly formatted datetime if input is valid else None
    """
    if 'h' not in timestamp:
        timestamp = '0h' + timestamp
    if 'm' not in timestamp:
        timestamp = timestamp.replace('h', 'h0m')
    if 's' not in timestamp:
        timestamp = timestamp + '0s'

    pattern = r'^(\d+h)?(\d+m)?(\d+s)?$'
    match = re.match(pattern, timestamp)
    if bool(match):
        return timestamp
    else:
        raise ValueError(f'{timestamp} was passed in, which does not fit XXhYYmZZs formatting')


class TimeUnit(StringEnum):
    """Enum class to represent units of time for the training process.

    Attributes:
        ITERATION (str): Iterations.
        EPOCH (str): Epochs.
        BATCH (str): Batches (i.e. number of optimization steps)
        SAMPLE (str): Samples.
        TOKEN (str): Tokens. Applicable for natural language processing (NLP) models.
        DURATION (str): Fraction of the training process complete, on ``[0.0, 1.0)``
    """
    ITERATION = 'iter'
    EPOCH = 'ep'
    BATCH = 'ba'
    SAMPLE = 'sp'
    TOKEN = 'tok'
    DURATION = 'dur'
    SECOND = 'sec'


# regex for parsing time string, matches timeunit and chars prior to unit as value
_TIME_STR_REGEX = re.compile(
    r'^(.+)(' + r'|'.join([fr'{time_unit.value}' for time_unit in TimeUnit]) + r')$',
    flags=re.IGNORECASE,
)

TValue = TypeVar('TValue', int, float)


class Time(Generic[TValue], Serializable):
    """Time represents static durations of training time in terms of a :class:`TimeUnit` enum.

    See the :doc:`Time Guide </trainer/time>` for more details on tracking time during training.

    To construct an instance of :class:`Time`, you can either:

    #. Use a value followed by a :class:`TimeUnit` enum or string. For example,

    >>> Time(5, TimeUnit.EPOCH)  # describes 5 epochs.
    Time(5, TimeUnit.EPOCH)
    >>> Time(30_000, "tok")  # describes 30,000 tokens.
    Time(30000, TimeUnit.TOKEN)
    >>> Time(0.5, "dur")  # describes 50% of the training process.
    Time(0.5, TimeUnit.DURATION)

    #. Use one of the helper methods. See:

    - :meth:`Time.from_epoch`
    - :meth:`Time.from_batch`
    - :meth:`Time.from_sample`
    - :meth:`Time.from_token`
    - :meth:`Time.from_duration`
    - :meth:`Time.from_timestring`.

    :class:`Time` supports addition and subtraction with other :class:`Time` instances that share the same
    :class:`TimeUnit`. For example:

    >>> Time(1, TimeUnit.EPOCH) + Time(2, TimeUnit.EPOCH)
    Time(3, TimeUnit.EPOCH)

    :class:`Time` supports multiplication. The multiplier must be either a number or have units of
    :attr:`TimeUnit.DURATION`. The multiplicand is scaled, and its units are kept.

    >>> Time(2, TimeUnit.EPOCH) * 0.5
    Time(1, TimeUnit.EPOCH)

    >>> Time(2, TimeUnit.EPOCH) * Time(0.5, TimeUnit.DURATION)
    Time(1, TimeUnit.EPOCH)


    :class:`Time` supports division. If the divisor is an instance of :class:`Time`, then it
    must have the same units as the dividend, and the result has units of :attr:`TimeUnit.DURATION`.
    For example:

    >>> Time(4, TimeUnit.EPOCH) / Time(2, TimeUnit.EPOCH)
    Time(2.0, TimeUnit.DURATION)

    If the divisor is number, then the dividend is scaled, and it keeps its units. For example:

    >>> Time(4, TimeUnit.EPOCH) / 2
    Time(2, TimeUnit.EPOCH)

    Args:
        value (int | float): The amount of time.
        unit (str | TimeUnit): The :class:`TimeUnit` for ``value``.
    """

    def __init__(
        self,
        value: TValue,
        unit: Union[str, TimeUnit],
    ):
        unit = TimeUnit(unit)
        if unit == TimeUnit.DURATION:
            value = cast(TValue, float(value))
        else:
            if not isinstance(value, int):
                raise TypeError(f'value {value} is of type {type(value)}. Units {unit} require integer values.')
        self._value, self._unit = value, TimeUnit(unit)

    @classmethod
    def from_iteration(cls, iteration: int) -> Time:
        """Create a :class:`Time` with units of :attr:`TimeUnit.ITERATION`.

        Equivalent to ``Time(iteration, TimeUnit.ITERATION)``.

        Args:
            iteration (int): Number of iterations.

        Returns:
            Time: :class:`Time` instance, in iterations.
        """
        return cls(iteration, TimeUnit.ITERATION)

    @classmethod
    def from_epoch(cls, epoch: int) -> Time:
        """Create a :class:`Time` with units of :attr:`TimeUnit.EPOCH`.

        Equivalent to ``Time(epoch, TimeUnit.EPOCH)``.

        Args:
            epoch (int): Number of epochs.

        Returns:
            Time: :class:`Time` instance, in epochs.
        """
        return cls(epoch, TimeUnit.EPOCH)

    @classmethod
    def from_batch(cls, batch: int) -> Time:
        """Create a :class:`Time` with units of :attr:`TimeUnit.BATCH`.

        Equivalent to ``Time(batch, TimeUnit.BATCH)``.

        Args:
            batch (int): Number of batches.

        Returns:
            Time: :class:`Time` instance, in batches.
        """
        return cls(batch, TimeUnit.BATCH)

    @classmethod
    def from_sample(cls, sample: int) -> Time:
        """Create a :class:`Time` with units of :attr:`TimeUnit.SAMPLE`.

        Equivalent to ``Time(sample, TimeUnit.SAMPLE)``.

        Args:
            sample (int): Number of samples.

        Returns:
            Time: :class:`Time` instance, in samples.
        """
        return cls(sample, TimeUnit.SAMPLE)

    @classmethod
    def from_token(cls, token: int) -> Time:
        """Create a :class:`Time` with units of :attr:`TimeUnit.TOKEN`.

        Equivalent to ``Time(sample, TimeUnit.TOKEN)``.

        Args:
            token (int): Number of tokens.

        Returns:
            Time: :class:`Time` instance, in tokens.
        """
        return cls(token, TimeUnit.TOKEN)

    @classmethod
    def from_duration(cls, duration: float) -> Time:
        """Create a :class:`Time` with units of :attr:`TimeUnit.DURATION`.

        Equivalent to ``Time(duration, TimeUnit.DURATION)``.

        Args:
            duration (float): Duration of the training process. Should be on ``[0, 1)``
                where ``0`` represents the beginning of the training process and ``1``
                represents a completed training process.

        Returns:
            Time: :class:`Time` instance, in duration.
        """
        return cls(duration, TimeUnit.DURATION)

    @classmethod
    def from_timedelta(cls, timestring: str) -> Time:
        """Create a :class:`Time` with units of :attr:`TimeUnit.SECOND`.

        Equivalent to ``Time(batch, TimeUnit.SECOND)``.

        Args:
            timestring (int): timedelta string in _h_m_s.

        Returns:
            Time: :class:`Time` instance, in seconds.
        """
        # Convert timestring to be strptime parsable
        verified_wct = verify_wct(timestring)
        time_struct = datetime.datetime.strptime(verified_wct, '%Hh%Mm%Ss')
        delta = datetime.timedelta(hours=time_struct.hour, minutes=time_struct.minute, seconds=time_struct.second)
        total_seconds = delta.total_seconds()
        return cls(int(total_seconds), TimeUnit.SECOND)

    @property
    def value(self) -> TValue:
        """The value of the time, as a number."""
        return self._value

    @property
    def unit(self) -> TimeUnit:
        """The unit of the time."""
        return self._unit

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.value}, {self.unit})'

    def __str__(self) -> str:
        return f'{self.value}{self.unit.value}'

    def to_timestring(self):
        """Get the time-string representation.

        For example:

        >>> Time(5, TimeUnit.EPOCH).to_timestring()
        '5ep'

        Returns:
            str: The time-string representation.
        """
        return str(self)

    def _parse(self, other: Union[int, float, Time, str]) -> Time:
        # parse ``other`` into a Time object
        return Time.from_input(other, self.unit)

    def _cmp(self, other: Union[int, float, Time, str]) -> int:
        # When doing comparisons, and other is an integer (or float), we can safely infer
        # the unit from self.unit
        # E.g. calls like this should be allowed: if batch < 42: do_something()
        # This eliminates the need to call .value everywhere
        if not isinstance(other, (int, float, Time, str)):
            return NotImplemented
        if isinstance(other, (int, float)):
            other = type(self)(other, self.unit)
        other = self._parse(other)
        if self.unit != other.unit:
            raise RuntimeError(f'Cannot compare {self} to {other} since they have different units.')
        if self.value < other.value:
            return -1
        if self.value == other.value:
            return 0
        assert self.value > other.value
        return 1

    def __eq__(self, other: Union[int, float, Time, str]):
        return self._cmp(other) == 0

    def __ne__(self, other: Union[int, float, Time, str]):
        return self._cmp(other) != 0

    def __lt__(self, other: Union[int, float, Time, str]):
        return self._cmp(other) < 0

    def __le__(self, other: Union[int, float, Time, str]):
        return self._cmp(other) <= 0

    def __gt__(self, other: Union[int, float, Time, str]):
        return self._cmp(other) > 0

    def __ge__(self, other: Union[int, float, Time, str]):
        return self._cmp(other) >= 0

    def __add__(self, other: Union[int, float, Time, str]) -> Time[TValue]:
        other = self._parse(other)
        if self.unit != other.unit:
            raise RuntimeError(f'Cannot add {self} to {other} since they have different units.')
        return Time(self.value + other.value, self.unit)

    def __radd__(self, other: Union[int, float, Time, str]) -> Time[TValue]:
        return self + other

    def __sub__(self, other: Union[int, float, Time, str]) -> Time[TValue]:
        other = self._parse(other)
        if self.unit != other.unit:
            raise RuntimeError(f'Cannot subtract {other} from {self} since they have different units.')
        return Time(self.value - other.value, self.unit)

    def __rsub__(self, other: Union[int, float, Time, str]) -> Time[TValue]:
        return (-self) + other

    def __neg__(self) -> Time[TValue]:
        return Time(cast(TValue, -self.value), self.unit)

    def __pos__(self) -> Time[TValue]:
        return Time(self.value, self.unit)

    def __int__(self):
        return int(self.value)

    def __float__(self):
        return float(self.value)

    def __truediv__(self, other: Union[int, float, Time, str]) -> Time[float]:
        if isinstance(other, (float, int)):
            return Time(type(self.value)(self.value / other), self.unit)
        other = self._parse(other)
        if self.unit != other.unit:
            raise RuntimeError(f'Cannot divide {self} by {other} since they have different units.')
        return Time(self.value / other.value, TimeUnit.DURATION)

    def __mod__(self, other: Union[int, float, Time, str]) -> Time[TValue]:
        other = self._parse(other)
        if self.unit != other.unit:
            raise RuntimeError(f'Cannot take mod of {self} by {other} since they have different units.')
        return Time(self.value % other.value, self.unit)

    def __mul__(self, other: Union[int, float, Time, str]):
        if isinstance(other, (float, int)):
            # Scale by the value.
            return Time(type(self.value)(self.value * other), self.unit)
        other = self._parse(other)
        if other.unit != TimeUnit.DURATION and self.unit != TimeUnit.DURATION:
            raise RuntimeError(f'Multiplication is supported only if one of the units is Duration')
        real_unit = self.unit if other.unit == TimeUnit.DURATION else other.unit
        real_type = float if real_unit == TimeUnit.DURATION else int
        return Time(real_type(self.value * other.value), real_unit)

    def __rmul__(self, other: Union[int, float, Time, str]):
        return self * other

    def __hash__(self):
        return hash((self.value, self.unit))

    @classmethod
    def from_input(
        cls,
        i: Union[str, int, float, 'Time'],
        default_int_unit: Optional[Union[TimeUnit, str]] = None,
    ) -> Time:
        """Parse a time input into a :class:`Time` instance.

        Args:
            i (str | int | Time): The time input.
            default_int_unit (TimeUnit, optional): The default unit to use if ``i`` is an integer

        >>> Time.from_input("5ep")
        Time(5, TimeUnit.EPOCH)
        >>> Time.from_input(5, TimeUnit.EPOCH)
        Time(5, TimeUnit.EPOCH)

        Returns:
            Time: An instance of :class:`Time`.
        """
        if isinstance(i, Time):
            return i

        if isinstance(i, str):
            return Time.from_timestring(i)

        if isinstance(i, int) or isinstance(i, float):
            if default_int_unit is None:
                raise RuntimeError('default_int_unit must be specified when constructing Time from an integer.')
            return Time(i, default_int_unit)

        raise RuntimeError(f'Cannot convert type {i} to {cls.__name__}')

    @classmethod
    def from_timestring(cls, timestring: str) -> Time:
        """Parse a time string into a :class:`Time` instance.

        A time string is a numerical value followed by the value of a :class:`TimeUnit` enum. For example:

        >>> Time.from_timestring("5ep")  # describes 5 epochs.
        Time(5, TimeUnit.EPOCH)
        >>> Time.from_timestring("3e4tok")  # describes 30,000 tokens.
        Time(30000, TimeUnit.TOKEN)
        >>> Time.from_timestring("0.5dur")  # describes 50% of the training process.
        Time(0.5, TimeUnit.DURATION)

        Returns:
            Time: An instance of :class:`Time`.
        """
        # Handle TimeDelta matching first
        try:
            return Time.from_timedelta(timestring)
        except ValueError:
            pass

        match = _TIME_STR_REGEX.findall(timestring)
        if len(match) != 1:
            raise ValueError(f'Invalid time string: {timestring}')
        match = match[0]
        match = [x for x in match if x != '']
        assert len(match) == 2, 'each match should have a number followed by the key'
        value = match[0]
        unit = TimeUnit(match[1])
        value = float(value)  # always parsing first as float b/c it could be scientific notation
        if unit != TimeUnit.DURATION:
            if int(value) != value:
                raise TypeError(f'value {value} is not an integer. Units {unit} require integer values.')
            value = int(value)
        return cls(value, unit)


class Timestamp(Serializable):
    """Timestamp represents a snapshot of the current training progress.

    The timestamp measures training progress in terms of iterations, epochs, batches, samples, tokens, and wall clock time.
    Timestamps are not updated in-place.

    See the :doc:`Time Guide </trainer/time>` for more details on tracking time during training.

    Args:
        iteration (int | Time[int], optional): The iteration.
        epoch (int | Time[int], optional): The epoch.
        batch (int | Time[int], optional): the batch.
        sample (int | Time[int], optional): The sample.
        token (int | Time[int], optional): The token.
        epoch_in_iteration (int | Time[int], optional): The epoch in the iteration.
        token_in_iteration (int | Time[int], optional): The token in the iteration.
        batch_in_epoch (int | Time[int], optional): The batch in the epoch.
        sample_in_epoch (int | Time[int], optional): The sample in the epoch.
        token_in_epoch (int | Time[int], optional): The token in the epoch.
        total_wct (datetime.timedelta, optional): The total wall-clock duration.
        iteration_wct (datetime.timedelta, optional): The wall-clock duration of the current iteration.
        epoch_wct (datetime.timedelta, optional): The wall-clock duration of the current epoch.
        batch_wct (datetime.timedelta, optional): The wall-clock duration of the last batch.
    """

    def __init__(
        self,
        iteration: Union[int, Time[int]] = 0,
        epoch: Union[int, Time[int]] = 0,
        batch: Union[int, Time[int]] = 0,
        sample: Union[int, Time[int]] = 0,
        token: Union[int, Time[int]] = 0,
        epoch_in_iteration: Union[int, Time[int]] = 0,
        token_in_iteration: Union[int, Time[int]] = 0,
        batch_in_epoch: Union[int, Time[int]] = 0,
        sample_in_epoch: Union[int, Time[int]] = 0,
        token_in_epoch: Union[int, Time[int]] = 0,
        total_wct: Optional[datetime.timedelta] = None,
        iteration_wct: Optional[datetime.timedelta] = None,
        epoch_wct: Optional[datetime.timedelta] = None,
        batch_wct: Optional[datetime.timedelta] = None,
    ):
        iteration = Time.from_input(iteration, TimeUnit.ITERATION)
        if iteration.unit != TimeUnit.ITERATION:
            raise ValueError(f'The `iteration` argument has units of {iteration.unit}; not {TimeUnit.ITERATION}.')
        self._iteration = iteration

        epoch = Time.from_input(epoch, TimeUnit.EPOCH)
        if epoch.unit != TimeUnit.EPOCH:
            raise ValueError(f'The `epoch` argument has units of {epoch.unit}; not {TimeUnit.EPOCH}.')
        self._epoch = epoch

        batch = Time.from_input(batch, TimeUnit.BATCH)
        if batch.unit != TimeUnit.BATCH:
            raise ValueError(f'The `batch` argument has units of {batch.unit}; not {TimeUnit.BATCH}.')
        self._batch = batch

        sample = Time.from_input(sample, TimeUnit.SAMPLE)
        if sample.unit != TimeUnit.SAMPLE:
            raise ValueError(f'The `sample` argument has units of {sample.unit}; not {TimeUnit.SAMPLE}.')
        self._sample = sample

        token = Time.from_input(token, TimeUnit.TOKEN)
        if token.unit != TimeUnit.TOKEN:
            raise ValueError(f'The `token` argument has units of {token.unit}; not {TimeUnit.TOKEN}.')
        self._token = token

        self._epoch_in_iteration = Time(0, TimeUnit.EPOCH)
        self.epoch_in_iteration = epoch_in_iteration

        token_in_iteration = Time.from_input(token_in_iteration, TimeUnit.TOKEN)
        if token_in_iteration.unit != TimeUnit.TOKEN:
            raise ValueError((
                f'The `token_in_iteration` argument has units of {token_in_iteration.unit}; '
                f'not {TimeUnit.TOKEN}.'
            ))
        self._token_in_iteration = token_in_iteration

        batch_in_epoch = Time.from_input(batch_in_epoch, TimeUnit.BATCH)
        if batch_in_epoch.unit != TimeUnit.BATCH:
            raise ValueError(
                (f'The `batch_in_epoch` argument has units of {batch_in_epoch.unit}; '
                 f'not {TimeUnit.BATCH}.'),
            )
        self._batch_in_epoch = batch_in_epoch

        sample_in_epoch = Time.from_input(sample_in_epoch, TimeUnit.SAMPLE)
        if sample_in_epoch.unit != TimeUnit.SAMPLE:
            raise ValueError(
                (f'The `sample_in_epoch` argument has units of {sample_in_epoch.unit}; '
                 f'not {TimeUnit.SAMPLE}.'),
            )
        self._sample_in_epoch = sample_in_epoch

        token_in_epoch = Time.from_input(token_in_epoch, TimeUnit.TOKEN)
        if token_in_epoch.unit != TimeUnit.TOKEN:
            raise ValueError(
                (f'The `token_in_epoch` argument has units of {token_in_epoch.unit}; '
                 f'not {TimeUnit.TOKEN}.'),
            )
        self._token_in_epoch = token_in_epoch

        if total_wct is None:
            total_wct = datetime.timedelta(seconds=0)
        self._total_wct = total_wct

        if iteration_wct is None:
            iteration_wct = datetime.timedelta(seconds=0)
        self._iteration_wct = iteration_wct

        if epoch_wct is None:
            epoch_wct = datetime.timedelta(seconds=0)
        self._epoch_wct = epoch_wct

        if batch_wct is None:
            batch_wct = datetime.timedelta(seconds=0)
        self._batch_wct = batch_wct

    def state_dict(self) -> dict[str, Any]:
        return {
            'iteration': self.iteration.value,
            'epoch': self.epoch.value,
            'batch': self.batch.value,
            'sample': self.sample.value,
            'token': self.token.value,
            'epoch_in_iteration': self.epoch_in_iteration.value,
            'token_in_iteration': self.token_in_iteration.value,
            'batch_in_epoch': self.batch_in_epoch.value,
            'sample_in_epoch': self.sample_in_epoch.value,
            'token_in_epoch': self.token_in_epoch.value,
            'total_wct': self.total_wct,
            'iteration_wct': self.iteration_wct,
            'epoch_wct': self.epoch_wct,
            'batch_wct': self.batch_wct,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self._epoch = Time(state['epoch'], TimeUnit.EPOCH)
        self._batch = Time(state['batch'], TimeUnit.BATCH)
        self._sample = Time(state['sample'], TimeUnit.SAMPLE)
        self._token = Time(state['token'], TimeUnit.TOKEN)
        self._batch_in_epoch = Time(state['batch_in_epoch'], TimeUnit.BATCH)
        self._sample_in_epoch = Time(state['sample_in_epoch'], TimeUnit.SAMPLE)
        self._token_in_epoch = Time(state['token_in_epoch'], TimeUnit.TOKEN)
        # Using conditional checks as not to break old checkpoints
        # Wall clock time tracking was added in composer v0.7.0
        if 'total_wct' in state:
            self._total_wct = state['total_wct']
        if 'epoch_wct' in state:
            self._epoch_wct = state['epoch_wct']
        if 'batch_wct' in state:
            self._batch_wct = state['batch_wct']
        # Iteration was added in composer v0.19.1
        if 'iteration' in state:
            self._iteration = Time(state['iteration'], TimeUnit.ITERATION)
        if 'epoch_in_iteration' in state:
            self.epoch_in_iteration = Time(state['epoch_in_iteration'], TimeUnit.EPOCH)
        if 'token_in_iteration' in state:
            self._token_in_iteration = Time(state['token_in_iteration'], TimeUnit.TOKEN)
        if 'iteration_wct' in state:
            self._iteration_wct = state['iteration_wct']

    @property
    def iteration(self) -> Time[int]:
        """The total iteration count."""
        return self._iteration

    @property
    def epoch(self) -> Time[int]:
        """The total epoch count."""
        return self._epoch

    @property
    def batch(self) -> Time[int]:
        """The total batch count."""
        return self._batch

    @property
    def sample(self) -> Time[int]:
        """The total sample count."""
        return self._sample

    @property
    def token(self) -> Time[int]:
        """The total token count."""
        return self._token

    @property
    def epoch_in_iteration(self) -> Time[int]:
        """The epoch count in the current iteration (resets at 0 at the beginning of every iteration)."""
        return self._epoch_in_iteration

    @epoch_in_iteration.setter
    def epoch_in_iteration(
        self,
        epoch_in_iteration: Union[int, Time[int]],  # pyright: ignore[reportPropertyTypeMismatch]
    ):
        """Sets epoch count in the current iteration."""
        epoch_in_iteration = Time.from_input(epoch_in_iteration, TimeUnit.EPOCH)
        if epoch_in_iteration.unit != TimeUnit.EPOCH:
            raise ValueError((
                f'The `epoch_in_iteration` argument has units of {epoch_in_iteration.unit}; '
                f'not {TimeUnit.EPOCH}.'
            ))
        self._epoch_in_iteration = epoch_in_iteration

    @property
    def token_in_iteration(self) -> Time[int]:
        """The token count in the current iteration (resets at 0 at the beginning of every iteration)."""
        return self._token_in_iteration

    @property
    def batch_in_epoch(self) -> Time[int]:
        """The batch count in the current epoch (resets at 0 at the beginning of every epoch)."""
        return self._batch_in_epoch

    @property
    def sample_in_epoch(self) -> Time[int]:
        """The sample count in the current epoch (resets at 0 at the beginning of every epoch)."""
        return self._sample_in_epoch

    @property
    def token_in_epoch(self) -> Time[int]:
        """The token count in the current epoch (resets at 0 at the beginning of every epoch)."""
        return self._token_in_epoch

    @property
    def total_wct(self) -> datetime.timedelta:
        """The wall-clock duration (in seconds) from the beginning of training."""
        return self._total_wct

    @property
    def iteration_wct(self) -> datetime.timedelta:
        """The wall-clock duration (in seconds) for the current iteration."""
        return self._iteration_wct

    @property
    def epoch_wct(self) -> datetime.timedelta:
        """The wall-clock duration (in seconds) for the current epoch."""
        return self._epoch_wct

    @property
    def batch_wct(self) -> datetime.timedelta:
        """The wall-clock duration (in seconds) for the last batch."""
        return self._batch_wct

    def get(self, unit: Union[str, TimeUnit]) -> Time[int]:
        """Returns the current time in the specified unit.

        Args:
            unit (str | TimeUnit): The desired unit.

        Returns:
            Time: The current time, in the specified unit.
        """
        unit = TimeUnit(unit)
        if unit == TimeUnit.ITERATION:
            return self.iteration
        if unit == TimeUnit.EPOCH:
            return self.epoch
        if unit == TimeUnit.BATCH:
            return self.batch
        if unit == TimeUnit.SAMPLE:
            return self.sample
        if unit == TimeUnit.TOKEN:
            return self.token
        if unit == TimeUnit.SECOND:
            return Time(int(self._total_wct.total_seconds()) if self._total_wct else 0, TimeUnit.SECOND)
        raise ValueError(f'Invalid unit: {unit}')

    def _parse(self, other: Union[int, float, Time, str]) -> Time:
        # parse ``other`` into a Time object
        if isinstance(other, Time):
            return other
        if isinstance(other, str):
            other_parsed = Time.from_timestring(other)
            return other_parsed

        raise TypeError(f'Cannot convert type {other} to {self.__class__.__name__}')

    def __eq__(self, other: Union[int, float, Time, str]):
        if not isinstance(other, (Time, Timestamp, str)):
            return NotImplemented
        if isinstance(other, Timestamp):
            return self.state_dict() == other.state_dict()
        other = self._parse(other)
        self_counter = self.get(other.unit)
        return self_counter == other

    def __ne__(self, other: Union[int, float, Time, str]):
        if not isinstance(other, (Time, Timestamp, str)):
            return NotImplemented
        if isinstance(other, Timestamp):
            return self.state_dict() != other.state_dict()
        other = self._parse(other)
        self_counter = self.get(other.unit)
        return self_counter != other

    def __lt__(self, other: Union[int, float, Time, str]):
        if not isinstance(other, (Time, str)):
            return NotImplemented
        other = self._parse(other)
        self_counter = self.get(other.unit)
        return self_counter < other

    def __le__(self, other: Union[int, float, Time, str]):
        if not isinstance(other, (Time, str)):
            return NotImplemented
        other = self._parse(other)
        self_counter = self.get(other.unit)
        return self_counter <= other

    def __gt__(self, other: Union[int, float, Time, str]):
        if not isinstance(other, (Time, str)):
            return NotImplemented
        other = self._parse(other)
        self_counter = self.get(other.unit)
        return self_counter > other

    def __ge__(self, other: Union[int, float, Time, str]):
        if not isinstance(other, (Time, str)):
            return NotImplemented
        other = self._parse(other)
        self_counter = self.get(other.unit)
        return self_counter >= other

    def to_next_batch(
        self,
        samples: Union[int, Time] = 0,
        tokens: Union[int, Time] = 0,
        duration: Optional[datetime.timedelta] = None,
    ):
        """Create a new :class:`.Timestamp`, advanced to the next batch.

        Equivalent to:

        .. testsetup::

            from composer.core.time import Timestamp
            import datetime

            timestamp = Timestamp()
            samples = 1
            tokens = 2
            duration = datetime.timedelta(seconds=0)

        .. doctest::

            >>> timestamp.copy(
            ...     batch=timestamp.batch + 1,
            ...     batch_in_epoch=timestamp.batch_in_epoch + 1,
            ...     sample=timestamp.sample + samples,
            ...     sample_in_epoch=timestamp.sample_in_epoch + samples,
            ...     token = timestamp.token + tokens,
            ...     token_in_epoch=timestamp.token_in_epoch + tokens,
            ...     total_wct=timestamp.total_wct + duration,
            ...     iteration_wct=timestamp.iteration_wct + duration,
            ...     epoch_wct=timestamp.epoch_wct + duration,
            ...     batch_wct=duration,
            ... )
            Timestamp(...)

        .. note::

            For accurate time tracking, when doing distributed training, the ``samples`` and ``tokens`` should be
            the total across all ranks for the given batch. This method will not accumulate these counts automatically.
            If per-rank sample and token counts are provided, these counts will differ across ranks, which could lead
            towards inconsistent behavior by :class:`.Algorithm` or :class:`.Callback` instances that use these counts.

        Args:
            samples (int | Time, optional): The number of samples trained in the batch. Defaults to 0.
            tokens (int | Time, optional): The number of tokens trained in the batch. Defaults to 0.
            duration (datetime.timedelta, optional): The duration to train the batch.
        """
        if duration is None:
            duration = datetime.timedelta(seconds=0)
        return self.copy(
            batch=self.batch + 1,
            batch_in_epoch=self.batch_in_epoch + 1,
            sample=self.sample + samples,
            sample_in_epoch=self.sample_in_epoch + samples,
            token=self.token + tokens,
            token_in_epoch=self.token_in_epoch + tokens,
            token_in_iteration=self.token_in_iteration + tokens,
            total_wct=self.total_wct + duration,
            iteration_wct=self.iteration_wct + duration,
            epoch_wct=self.epoch_wct + duration,
            batch_wct=duration,
        )

    def to_next_epoch(
        self,
        duration: Optional[datetime.timedelta] = None,
    ):
        """Create a new :class:`.Timestamp`, advanced to the next epoch.

        Equivalent to:

        .. testsetup::

            from composer.core.time import Timestamp
            import datetime

            timestamp = Timestamp()
            duration = datetime.timedelta(seconds=0)

        .. doctest::

            >>> timestamp.copy(
            ...     epoch=timestamp.epoch + 1,
            ...     epoch_in_iteration=timestamp.epoch_in_iteration + 1,
            ...     token_in_iteration=timestamp.token_in_iteration + tokens,
            ...     batch_in_epoch=0,
            ...     sample_in_epoch=0,
            ...     token_in_epoch=0,
            ...     total_wct=timestamp.total_wct + duration,
            ...     iteration_wct=timestamp.iteration_wct + duration,
            ...     epoch_wct=datetime.timedelta(seconds=0),
            ...     batch_wct=datetime.timedelta(seconds=0),
            ... )
            Timestamp(...)

        Args:
            tokens (int | Time, optional): The number of tokens trained in the batch. Defaults to 0.
            duration (datetime.timedelta, optional): The duration to train the batch.

        """
        if duration is None:
            duration = datetime.timedelta(seconds=0)
        return self.copy(
            epoch=self.epoch + 1,
            epoch_in_iteration=self.epoch_in_iteration + 1,
            batch_in_epoch=0,
            sample_in_epoch=0,
            token_in_epoch=0,
            total_wct=self.total_wct + duration,
            iteration_wct=self.iteration_wct + duration,
            epoch_wct=datetime.timedelta(seconds=0),
            batch_wct=datetime.timedelta(seconds=0),
        )

    def to_next_iteration(
        self,
        duration: Optional[datetime.timedelta] = None,
    ):
        """Create a new :class:`.Timestamp`, advanced to the next iteration.

        Equivalent to:

        .. testsetup::

            from composer.core.time import Timestamp
            import datetime

            timestamp = Timestamp()

        .. doctest::

            >>> timestamp.copy(
            ...     iteration=timestamp.iteration + 1,
            ...     epoch_in_iteration=0,
            ...     token_in_iteration=0,
            ...     batch_in_epoch=0,
            ...     sample_in_epoch=0,
            ...     token_in_epoch=0,
            ...     total_wct=timestamp.total_wct + duration,
            ...     iteration_wct=datetime.timedelta(seconds=0),
            ...     epoch_wct=datetime.timedelta(seconds=0),
            ...     batch_wct=datetime.timedelta(seconds=0),
            ... )
            Timestamp(...)

        """
        if duration is None:
            duration = datetime.timedelta(seconds=0)
        return self.copy(
            iteration=self.iteration + 1,
            epoch_in_iteration=0,
            token_in_iteration=0,
            batch_in_epoch=0,
            sample_in_epoch=0,
            token_in_epoch=0,
            total_wct=self.total_wct + duration,
            iteration_wct=datetime.timedelta(seconds=0),
            epoch_wct=datetime.timedelta(seconds=0),
            batch_wct=datetime.timedelta(seconds=0),
        )

    def copy(
        self,
        iteration: Optional[Union[int, Time[int]]] = None,
        epoch: Optional[Union[int, Time[int]]] = None,
        batch: Optional[Union[int, Time[int]]] = None,
        sample: Optional[Union[int, Time[int]]] = None,
        token: Optional[Union[int, Time[int]]] = None,
        epoch_in_iteration: Optional[Union[int, Time[int]]] = None,
        token_in_iteration: Optional[Union[int, Time[int]]] = None,
        batch_in_epoch: Optional[Union[int, Time[int]]] = None,
        sample_in_epoch: Optional[Union[int, Time[int]]] = None,
        token_in_epoch: Optional[Union[int, Time[int]]] = None,
        total_wct: Optional[datetime.timedelta] = None,
        iteration_wct: Optional[datetime.timedelta] = None,
        epoch_wct: Optional[datetime.timedelta] = None,
        batch_wct: Optional[datetime.timedelta] = None,
    ) -> Timestamp:
        """Create a copy of the timestamp.

        Any specified values will override the existing values in the returned copy.

        Args:
            iteration (int | Time[int], optional): The iteration.
            epoch (int | Time[int], optional): The epoch.
            batch (int | Time[int], optional): the batch.
            sample (int | Time[int], optional): The sample.
            token (int | Time[int], optional): The token.
            epoch_in_iteration (int | Time[int], optional): The epoch in the iteration.
            token_in_iteration (int | Time[int], optional): The token in the iteration.
            batch_in_epoch (int | Time[int], optional): The batch in the epoch.
            sample_in_epoch (int | Time[int], optional): The sample in the epoch.
            token_in_epoch (int | Time[int], optional): The token in the epoch.
            total_wct (datetime.timedelta, optional): The elapsed duration from the beginning of training.
            iteration_wct (datetime.timedelta, optional): The wall-clock duration of the current iteration.
            epoch_wct (datetime.timedelta, optional): The wall-clock duration of the current epoch.
            batch_wct (datetime.timedelta, optional): The wall-clock duration of the last batch.

        Returns:
            Timestamp: A new timestamp instance, created from a copy, but with any specified values
                overriding the existing values.
        """
        return Timestamp(
            iteration=iteration if iteration is not None else self.iteration,
            epoch=epoch if epoch is not None else self.epoch,
            batch=batch if batch is not None else self.batch,
            sample=sample if sample is not None else self.sample,
            token=token if token is not None else self.token,
            epoch_in_iteration=epoch_in_iteration if epoch_in_iteration is not None else self.epoch_in_iteration,
            token_in_iteration=token_in_iteration if token_in_iteration is not None else self.token_in_iteration,
            batch_in_epoch=batch_in_epoch if batch_in_epoch is not None else self.batch_in_epoch,
            sample_in_epoch=sample_in_epoch if sample_in_epoch is not None else self.sample_in_epoch,
            token_in_epoch=token_in_epoch if token_in_epoch is not None else self.token_in_epoch,
            total_wct=total_wct if total_wct is not None else self.total_wct,
            iteration_wct=iteration_wct if iteration_wct is not None else self.iteration_wct,
            epoch_wct=epoch_wct if epoch_wct is not None else self.epoch_wct,
            batch_wct=batch_wct if batch_wct is not None else self.batch_wct,
        )

    def __repr__(self) -> str:
        return (
            f'Timestamp('
            f'iteration={int(self.iteration)}, '
            f'epoch={int(self.epoch)}, '
            f'batch={int(self.batch)}, '
            f'sample={int(self.sample)}, '
            f'token={int(self.token)}, '
            f'epoch_in_iteration={int(self.epoch_in_iteration)}, '
            f'token_in_iteration={int(self.token_in_iteration)}, '
            f'batch_in_epoch={int(self.batch_in_epoch)}, '
            f'sample_in_epoch={int(self.sample_in_epoch)}, '
            f'token_in_epoch={int(self.token_in_epoch)}, '
            f'total_wct={repr(self.total_wct)}, '
            f'iteration_wct={repr(self.iteration_wct)}, '
            f'epoch_wct={repr(self.epoch_wct)}, '
            f'batch_wct={repr(self.batch_wct)}'
            ')'
        )


def ensure_time(maybe_time: Union[Time, str, int], int_unit: Union[TimeUnit, str]) -> Time:
    """Ensure ``maybe_time`` is an instance of :class:`.Time`.

    Args:
        maybe_time (Time | str): A time string, integer, or instance of :class:`.Time`.
        int_unit (TimeUnit | str): The unit to use if ``maybe_time`` is an integer

    Returns:
        Time: An instance of :class:`.Time`.
    """
    time_obj = Time.from_input(maybe_time, int_unit)
    return time_obj
