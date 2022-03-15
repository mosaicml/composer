# Copyright 2021 MosaicML. All Rights Reserved.

"""Utilities to track training progress in terms of epochs, batches, samples, and tokens.

Callbacks, algorithms, and schedulers can use the current training time to fire at certain points in the training
process.

The :class:`~.time.Timer` class tracks the total number of epochs, batches, samples, and tokens. The trainer is
responsible for updating it at the end of every epoch and batch.  There is only one instance of the
:class:`~.time.Timer`, which is attached to the :class:`~.state.State`.

The :class:`~.time.Time` class represents static durations of training time or points in the training process in terms
of a specific :class:`~.time.TimeUnit` enum. This class supports comparisons, arithmetic, and conversions.

See the :doc:`Time Guide </trainer/time>` for more details on tracking time during training.
"""
from __future__ import annotations

import re
import textwrap
import warnings
from typing import TYPE_CHECKING, Generic, NamedTuple, TypeVar, Union, cast

from composer.core.serializable import Serializable
from composer.utils.string_enum import StringEnum

if TYPE_CHECKING:
    from composer.core.types import StateDict

__all__ = ["TimeUnit", "Time", "Timer", "Timestamp"]


class TimeUnit(StringEnum):
    """Enum class to represent units of time for the training process.

    Attributes:
        EPOCH (str): Epochs.
        BATCH (str): Batches (i.e. number of optimization steps)
        SAMPLE (str): Samples.
        TOKEN (str): Tokens. Applicable for natural language processing (NLP) models.
        DURATION (str): Fraction of the training process complete, on ``[0.0, 1.0)``
    """
    EPOCH = "ep"
    BATCH = "ba"
    SAMPLE = "sp"
    TOKEN = "tok"
    DURATION = "dur"


# regex for parsing integers / decimals / scientific notation
_NUM_REGEX = r'-?[\d.]+(?:e-?\d+)?'

# regex for parsing a time string.
_TIME_STR_REGEX = re.compile(r'^(?:' + r'|'.join(fr"(?:({_NUM_REGEX})({time_unit.value}))" for time_unit in TimeUnit) +
                             r')$',
                             flags=re.IGNORECASE)

TValue = TypeVar("TValue", int, float)


class Time(Generic[TValue]):
    """Time represents static durations of training time or points in the training process in terms of a
    :class:`TimeUnit` enum (epochs, batches, samples, tokens, or duration).

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
        value (int or float): The amount of time.
        unit (str or TimeUnit): The :class:`TimeUnit` for ``value``.
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
                raise TypeError(f"value {value} is of type {type(value)}. Units {unit} require integer values.")
        self._value, self._unit = value, TimeUnit(unit)

    @classmethod
    def from_epoch(cls, epoch: int) -> Time:
        """Create a :class:`Time` with units of :attr:`TimeUnit.EPOCH`. Equivalent to ``Time(epoch, TimeUnit.EPOCH)``.

        Args:
            epoch (int): Number of epochs.

        Returns:
            Time: :class:`Time` instance, in epochs.
        """
        return cls(epoch, TimeUnit.EPOCH)

    @classmethod
    def from_batch(cls, batch: int) -> Time:
        """Create a :class:`Time` with units of :attr:`TimeUnit.BATCH`. Equivalent to ``Time(batch, TimeUnit.BATCH)``.

        Args:
            batch (int): Number of batches.

        Returns:
            Time: :class:`Time` instance, in batches.
        """
        return cls(batch, TimeUnit.BATCH)

    @classmethod
    def from_sample(cls, sample: int) -> Time:
        """Create a :class:`Time` with units of :attr:`TimeUnit.SAMPLE`. Equivalent to ``Time(sample,
        TimeUnit.SAMPLE)``.

        Args:
            sample (int): Number of samples.

        Returns:
            Time: :class:`Time` instance, in samples.
        """
        return cls(sample, TimeUnit.SAMPLE)

    @classmethod
    def from_token(cls, token: int) -> Time:
        """Create a :class:`Time` with units of :attr:`TimeUnit.TOKEN`. Equivalent to ``Time(sample, TimeUnit.TOKEN)``.

        Args:
            token (int): Number of tokens.

        Returns:
            Time: :class:`Time` instance, in tokens.
        """
        return cls(token, TimeUnit.TOKEN)

    @classmethod
    def from_duration(cls, duration: float) -> Time:
        """Create a :class:`Time` with units of :attr:`TimeUnit.DURATION`. Equivalent to ``Time(duration,
        TimeUnit.DURATION)``.

        Args:
            duration (float): Duration of the training process. Should be on ``[0, 1)``
                where ``0`` represents the beginning of the training process and ``1``
                represents a completed training process.

        Returns:
            Time: :class:`Time` instance, in duration.
        """
        return cls(duration, TimeUnit.DURATION)

    @property
    def value(self) -> TValue:
        """The value of the time, as a number."""
        return self._value

    @property
    def unit(self) -> TimeUnit:
        """The unit of the time."""
        return self._unit

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.value}, {self.unit})"

    def __str__(self) -> str:
        return f"{self.value}{self.unit.value}"

    def to_timestring(self):
        """Get the time-string representation.

        For example:

        >>> Time(5, TimeUnit.EPOCH).to_timestring()
        '5ep'

        Returns:
            str: The time-string representation.
        """
        return str(self)

    def _parse(self, other: object) -> Time:
        # parse ``other`` into a Time object
        if isinstance(other, Time):
            return other
        if isinstance(other, str):
            other_parsed = Time.from_timestring(other)
            warnings.warn(
                textwrap.dedent(f"""\
                    TimeImplicitStringConversion:
                    Implicitly converting {other} to {other_parsed}.
                    To fix this warning, replace {other} with {other_parsed}."""))
            return other_parsed

        raise TypeError(f"Cannot convert type {other} to {self.__class__.__name__}")

    def _cmp(self, other: object) -> int:
        # When doing comparisions, and other is an integer (or float), we can safely infer
        # the unit from self.unit
        # E.g. calls like this should be allowed: if batch < 42: do_something()
        # This eliminates the need to call .value everywhere
        if not isinstance(other, (int, float, Time, str)):
            return NotImplemented
        if isinstance(other, (int, float)):
            other = type(self)(other, self.unit)
        other = self._parse(other)
        if self.unit != other.unit:
            raise RuntimeError(f"Cannot compare {self} to {other} since they have different units.")
        if self.value < other.value:
            return -1
        if self.value == other.value:
            return 0
        assert self.value > other.value
        return 1

    def __eq__(self, other: object):
        return self._cmp(other) == 0

    def __ne__(self, other: object):
        return self._cmp(other) != 0

    def __lt__(self, other: object):
        return self._cmp(other) < 0

    def __le__(self, other: object):
        return self._cmp(other) <= 0

    def __gt__(self, other: object):
        return self._cmp(other) > 0

    def __ge__(self, other: object):
        return self._cmp(other) >= 0

    def __add__(self, other: object) -> Time[TValue]:
        other = self._parse(other)
        if self.unit != other.unit:
            raise RuntimeError(f"Cannot add {self} to {other} since they have different units.")
        return Time(self.value + other.value, self.unit)

    def __radd__(self, other: object) -> Time[TValue]:
        return self + other

    def __sub__(self, other: object) -> Time[TValue]:
        other = self._parse(other)
        if self.unit != other.unit:
            raise RuntimeError(f"Cannot subtract {other} from {self} since they have different units.")
        return Time(self.value - other.value, self.unit)

    def __rsub__(self, other: object) -> Time[TValue]:
        return (-self) + other

    def __neg__(self) -> Time[TValue]:
        return Time(cast(TValue, -self.value), self.unit)

    def __pos__(self) -> Time[TValue]:
        return Time(self.value, self.unit)

    def __int__(self):
        return int(self.value)

    def __float__(self):
        return float(self.value)

    def __truediv__(self, other: object) -> Time[float]:
        if isinstance(other, (float, int)):
            return Time(type(self.value)(self.value / other), self.unit)
        other = self._parse(other)
        if self.unit != other.unit:
            raise RuntimeError(f"Cannot divide {self} by {other} since they have different units.")
        return Time(self.value / other.value, TimeUnit.DURATION)

    def __mul__(self, other: object):
        if isinstance(other, (float, int)):
            # Scale by the value.
            return Time(type(self.value)(self.value * other), self.unit)
        other = self._parse(other)
        if other.unit != TimeUnit.DURATION and self.unit != TimeUnit.DURATION:
            raise RuntimeError(f"Multiplication is supported only if one of the units is Duration")
        real_unit = self.unit if other.unit == TimeUnit.DURATION else other.unit
        real_type = float if real_unit == TimeUnit.DURATION else int
        return Time(real_type(self.value * other.value), real_unit)

    def __rmul__(self, other: object):
        return self * other

    def __hash__(self):
        return hash((self.value, self.unit))

    @classmethod
    def from_timestring(cls, timestring: str) -> Time:
        """Parse a time string into a :class:`Time` instance. A time string is a numerical value followed by the value
        of a :class:`TimeUnit` enum. For example:

        >>> Time.from_timestring("5ep")  # describes 5 epochs.
        Time(5, TimeUnit.EPOCH)
        >>> Time.from_timestring("3e4tok")  # describes 30,000 tokens.
        Time(30000, TimeUnit.TOKEN)
        >>> Time.from_timestring("0.5dur")  # describes 50% of the training process.
        Time(0.5, TimeUnit.DURATION)

        Returns:
            Time: An instance of :class:`Time`.
        """
        match = _TIME_STR_REGEX.findall(timestring)
        if len(match) != 1:
            raise ValueError(f"Invalid time string: {timestring}")
        match = match[0]
        match = [x for x in match if x != '']
        assert len(match) == 2, "each match should have a number followed by the key"
        value = match[0]
        unit = TimeUnit(match[1])
        value = float(value)  # always parsing first as float b/c it could be scientific notation
        if unit != TimeUnit.DURATION:
            value = int(value)
        return cls(value, unit)


class Timer(Serializable):
    """Timer tracks the current training progress, in terms of epochs, batches, samples, and tokens.

    See the :doc:`Time Guide </trainer/time>` for more details on tracking time during training.

    .. note::

        An instance of this class is automatically constructed by the :class:`~composer.core.state.State` constructor.
        A user need not instantiate this class.
    """

    def __init__(self):
        self._epoch = Time(0, TimeUnit.EPOCH)
        self._batch = Time(0, TimeUnit.BATCH)
        self._sample = Time(0, TimeUnit.SAMPLE)
        self._token = Time(0, TimeUnit.TOKEN)
        self._batch_in_epoch = Time(0, TimeUnit.BATCH)
        self._sample_in_epoch = Time(0, TimeUnit.SAMPLE)
        self._token_in_epoch = Time(0, TimeUnit.TOKEN)

    def state_dict(self) -> StateDict:
        return {
            "epoch": self.epoch.value,
            "batch": self.batch.value,
            "sample": self.sample.value,
            "token": self.token.value,
            "batch_in_epoch": self.batch_in_epoch.value,
            "sample_in_epoch": self.sample_in_epoch.value,
            "token_in_epoch": self.token_in_epoch.value,
        }

    def load_state_dict(self, state: StateDict) -> None:
        self._epoch = Time(state["epoch"], TimeUnit.EPOCH)
        self._batch = Time(state["batch"], TimeUnit.BATCH)
        self._sample = Time(state["sample"], TimeUnit.SAMPLE)
        self._token = Time(state["token"], TimeUnit.TOKEN)
        self._batch_in_epoch = Time(state["batch_in_epoch"], TimeUnit.BATCH)
        self._sample_in_epoch = Time(state["sample_in_epoch"], TimeUnit.SAMPLE)
        self._token_in_epoch = Time(state["token_in_epoch"], TimeUnit.TOKEN)

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

    def get(self, unit: Union[str, TimeUnit]) -> Time[int]:
        """Returns the current time in the specified unit.

        Args:
            unit (str or TimeUnit): The desired unit.

        Returns:
            Time: The current time, in the specified unit.
        """
        unit = TimeUnit(unit)
        if unit == TimeUnit.EPOCH:
            return self.epoch
        if unit == TimeUnit.BATCH:
            return self.batch
        if unit == TimeUnit.SAMPLE:
            return self.sample
        if unit == TimeUnit.TOKEN:
            return self.token
        raise ValueError(f"Invalid unit: {unit}")

    def on_batch_complete(self, samples: Union[int, Time] = 0, tokens: Union[int, Time] = 0):
        """Called by the trainer at the end of every optimization batch.

        .. note::

            For accurate time tracking, the trainer is responsible for accumulating the total number of
            samples and/or tokens trained across all ranks before invoking this function.

        Args:
            samples (int or Time, optional): The number of samples trained in the batch. Defaults to 0.
            tokens (int or Time, optional): The number of tokens trained in the batch. Defaults to 0.
        """
        self._batch += Time(1, TimeUnit.BATCH)
        self._batch_in_epoch += Time(1, TimeUnit.BATCH)
        if isinstance(samples, int):
            samples = Time(samples, TimeUnit.SAMPLE)
        if isinstance(tokens, int):
            tokens = Time(tokens, TimeUnit.TOKEN)
        self._sample += samples
        self._sample_in_epoch += samples
        self._token += tokens
        self._token_in_epoch += tokens

    def on_epoch_complete(self):
        """Called by the trainer at the end of an epoch."""
        self._epoch += Time(1, TimeUnit.EPOCH)
        self._batch_in_epoch = Time(0, TimeUnit.BATCH)
        self._sample_in_epoch = Time(0, TimeUnit.SAMPLE)
        self._token_in_epoch = Time(0, TimeUnit.TOKEN)

    def _parse(self, other: object) -> Time:
        # parse ``other`` into a Time object
        if isinstance(other, Time):
            return other
        if isinstance(other, str):
            other_parsed = Time.from_timestring(other)
            warnings.warn(
                textwrap.dedent(f"""\
                    TimeImplicitStringConversion:
                    Implicitly converting {other} to {other_parsed}.
                    To fix this warning, replace {other} with {other_parsed}."""))
            return other_parsed

        raise TypeError(f"Cannot convert type {other} to {self.__class__.__name__}")

    def __eq__(self, other: object):
        if not isinstance(other, (Time, Timer, str)):
            return NotImplemented
        if isinstance(other, Timer):
            return self.state_dict() == other.state_dict()
        other = self._parse(other)
        self_counter = self.get(other.unit)
        return self_counter == other

    def __ne__(self, other: object):
        if not isinstance(other, (Time, Timer, str)):
            return NotImplemented
        if isinstance(other, Timer):
            return self.state_dict() != other.state_dict()
        other = self._parse(other)
        self_counter = self.get(other.unit)
        return self_counter != other

    def __lt__(self, other: object):
        if not isinstance(other, (Time, str)):
            return NotImplemented
        other = self._parse(other)
        self_counter = self.get(other.unit)
        return self_counter < other

    def __le__(self, other: object):
        if not isinstance(other, (Time, str)):
            return NotImplemented
        other = self._parse(other)
        self_counter = self.get(other.unit)
        return self_counter <= other

    def __gt__(self, other: object):
        if not isinstance(other, (Time, str)):
            return NotImplemented
        other = self._parse(other)
        self_counter = self.get(other.unit)
        return self_counter > other

    def __ge__(self, other: object):
        if not isinstance(other, (Time, str)):
            return NotImplemented
        other = self._parse(other)
        self_counter = self.get(other.unit)
        return self_counter >= other

    def get_timestamp(self):
        """Returns a snapshot of the current time.

        Unlike the :class:`Timer`, the values in a :class:`Timestamp` are a snapshot and are NOT incremented as
        training progresses.

        Returns:
            Timestamp: A snapshot of the current training time.
        """
        return Timestamp(
            epoch=self.epoch,
            batch=self.batch,
            batch_in_epoch=self.batch_in_epoch,
            sample=self.sample,
            sample_in_epoch=self.sample_in_epoch,
            token=self.token,
            token_in_epoch=self.token_in_epoch,
        )


class Timestamp(NamedTuple):
    """Timestamp represents a snapshot of :class:`Timer`.

    It is returned from a call to :meth:`Timer.get_timestamp`.

    Unlike the :class:`Timer`, the values in a :class:`Timestamp` are a snapshot and are NOT incremented as training
    progresses.

    See the :doc:`Time Guide </trainer/time>` for more details on tracking time during training.

    .. note::

        :class:`Timestamp` should not be instantiated directly; instead use :meth:`Timer.get_timestamp`.

    Attributes:
        epoch (Time[int]): The total epoch count when the :class`Timestamp` was generated.
        batch (Time[int]): The total batch count when the :class`Timestamp` was generated.
        batch_in_epoch (Time[int]): The batch count in the epoch when the :class`Timestamp` was generated.
        sample (Time[int]): The total sample count when the :class`Timestamp` was generated.
        sample_in_epoch (Time[int]): The sample count in the epoch when the :class`Timestamp` was generated.
        token (Time[int]): The total token count when the :class`Timestamp` was generated.
        token_in_epoch (Time[int]): The token count in the epoch when the :class`Timestamp` was generated.
    """
    epoch: Time[int]
    batch: Time[int]
    batch_in_epoch: Time[int]
    sample: Time[int]
    sample_in_epoch: Time[int]
    token: Time[int]
    token_in_epoch: Time[int]
