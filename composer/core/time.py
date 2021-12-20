# Copyright 2021 MosaicML. All Rights Reserved.

from __future__ import annotations

import re
import textwrap
import warnings
from typing import TYPE_CHECKING, Optional, Union

from composer.core.serializable import Serializable
from composer.utils.string_enum import StringEnum

if TYPE_CHECKING:
    from composer.core.types import StateDict


class TimeUnit(StringEnum):
    """Units of time for the training process.

    Attributes:
        EPOCH (str): Epochs.
        BATCH (str): Batchs (i.e. number of optimization steps)
        SAMPLE (str): Samples.
        TOKEN (str): Tokens. Applicable for natural language processing (NLP) models.
        DURATION (str): Fraction of the training process complete, on ``[0.0, 1.0)``
    """
    EPOCH = "ep"
    BATCH = "ba"
    SAMPLE = "sp"
    TOKEN = "tok"
    DURATION = "dur"


_NUM_REGEX = r'-?[\d.]+(?:e-?\d+)?'  # regex for parsing integers / decimals / scientific notation

# regex for parsing a time string.
_TIME_STR_REGEX = re.compile(r'^(?:' + r'|'.join(fr"(?:({_NUM_REGEX})({time_unit.value}))" for time_unit in TimeUnit) +
                             r')$',
                             flags=re.IGNORECASE)


class Time:
    """Time represents static durations of training time or points in the training process in terms of a 
    :class:`TimeUnit` enum (epochs, batches, samples, tokens, or duration).

    To construct an instance of :class:`Time`, you can either:
        
        #. Use a value followed by a :class:`TimeUnit` enum or string. For example,

            >>> Time(5, TimeUnit.EPOCH)  # describes 5 epochs.
            >>> Time(3e4, "tok")  # describes 30,000 tokens.
            >>> Time(0.5, "dur")  # describes 50% of the training process.

        #. Use one of the helper methods. See:

            - :meth:`Time.from_epoch`
            - :meth:`Time.from_batch`
            - :meth:`Time.from_sample`
            - :meth:`Time.from_token`
            - :meth:`Time.from_duration`
            - :meth:`Time.from_timestring`.

    :class:`Time` supports addition and subtraction with other :class:`Time` instances that share the same
    :class:`TimeUnit`. For example:

    >>> Time(1, TimeUnit.EPOCH) + Time(2, TimeUnit.EPOCH) == Time(3, TimeUnit.EPOCH)

    :class:`Time` supports multiplication. The multiplier must be either a number or have units of
    :attr:`TimeUnit.DURATION`. The multiplicand is scaled, and its units are kept.

    >>> Time(2, TimeUnit.EPOCH) * 0.5 == Time(1, TimeUnit.EPOCH)
    >>> Time(2, TimeUnit.EPOCH) * Time(0.5, TimeUnit.DURATION) == Time(1, TimeUnit.EPOCH)


    :class:`Time` supports division. If the divisor is an instance of :class:`Time`, then it
    must have the same units as the dividend, and the result has units of :attr:`TimeUnit.DURATION`.
    For example:

    >>> Time(4, TimeUnit.EPOCH) / Time(2, TimeUnit.EPOCH) == Time(2.0, TimeUnit.DURATION)

    If the divisor is number, then the dividend is scaled, and it keeps its units. For example:

    >>> Time(4, TimeUnit.EPOCH) / 2 == Time(2, TimeUnit.EPOCH)

    Args:
        value (int or float): The amount of time.
        unit (str or TimeUnit): The :class:`TimeUnit` for ``value``.
    """

    def __init__(
        self,
        value: Union[int, float],
        unit: Union[str, TimeUnit],
    ):
        unit = TimeUnit(unit)
        if unit == TimeUnit.DURATION:
            value = float(value)
        else:
            if not isinstance(value, int):
                raise TypeError(f"value {value} is of type {type(value)}. Units {unit} require integer values.")
        self._value, self._unit = value, TimeUnit(unit)

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

    @property
    def value(self) -> Union[int, float]:
        """The value of the time, as a number."""
        return self._value

    @property
    def unit(self) -> TimeUnit:
        """The unit of the time."""
        return self._unit

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.value}, {self.unit})"

    def _parse(self, other: object) -> Time:
        # parse ``other`` into a Time object
        if isinstance(other, Time):
            return other
        if isinstance(other, str):
            other_parsed = Time.from_timestring(other)
            warnings.warn(
                textwrap.dedent(f"""TimeImplicitStringConversion:
                Implicitly converting {other} to {other_parsed}.
                To fix this warning, replace {other} with {other_parsed}."""))
            return other_parsed

        raise NotImplementedError(f"Cannot convert type {other} to {self.__class__.__name__}")

    def _cmp(self, other: object) -> int:
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

    def __add__(self, other: object):
        other = self._parse(other)
        if self.unit != other.unit:
            raise RuntimeError(f"Cannot add {self} to {other} since they have different units.")
        return Time(self.value + other.value, self.unit)

    def __radd__(self, other: object):
        return self + other

    def __sub__(self, other: object):
        other = self._parse(other)
        if self.unit != other.unit:
            raise RuntimeError(f"Cannot subtract {other} from {self} since they have different units.")
        return Time(self.value - other.value, self.unit)

    def __rsub__(self, other: object):
        return (-self) + other

    def __neg__(self):
        return Time(-self.value, self.unit)

    def __pos__(self):
        return Time(self.value, self.unit)

    def __int__(self):
        return int(self.value)

    def __float__(self):
        return float(self.value)

    def __truediv__(self, other: object):
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

    @classmethod
    def from_timestring(cls, timestring: str) -> Time:
        """Parse a time string into a :class:`Time` instance.
        A time string is a numerical value followed by the value of a :class:`TimeUnit` enum. For example:

        >>> Time("5ep")  # describes 5 epochs.
        >>> Time("3e4tok")  # describes 30,000 tokens.
        >>> Time("0.5dur")  # describes 50% of the training process.

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
        return Time(value, unit)

    def convert(
        self,
        unit: Union[TimeUnit, str],
        *,
        batch_size: Optional[int] = None,
        drop_last: Optional[bool] = None,
        dataset_num_samples: Optional[int] = None,
        dataset_num_tokens: Optional[int] = None,
        max_training_duration: Optional[Union[str, Time]] = None,
    ) -> Time:
        r"""Convert a :class:`Time` instance into the specified ``unit``.

        Parameter ``unit`` is always required. The following table lists the additional required parameters
        to perform the conversion:

        +-----------------------------------------------------+-------------------------------------------------+-----------------------------------------------------------+-----------------------------------------------------------+-----------------------------------+-----------------------------+
        | From Unit |:arrow_down:| \\ To Unit |:arrow_right:| | :attr:`~TimeUnit.EPOCH`                         | :attr:`~TimeUnit.BATCH`                                   | :attr:`~TimeUnit.SAMPLE`                                  | :attr:`~TimeUnit.TOKEN`           | :attr:`~TimeUnit.DURATION`  |
        +-----------------------------------------------------+-------------------------------------------------+-----------------------------------------------------------+-----------------------------------------------------------+-----------------------------------+-----------------------------+
        | :attr:`~TimeUnit.EPOCH`                             | No required parameters.                         | - ``batch_size``                                          | - ``drop_last``                                           | - ``dataset_num_tokens``          | - ``max_training_duration`` |
        |                                                     |                                                 | - ``drop_last``                                           | - ``dataset_num_samples``                                 | - ``drop_last`` must be ``False`` |                             |
        |                                                     |                                                 | - ``dataset_num_samples``                                 | - ``batch_size`` (if ``drop_last`` is ``True``)           |                                   |                             |
        +-----------------------------------------------------+-------------------------------------------------+-----------------------------------------------------------+-----------------------------------------------------------+-----------------------------------+-----------------------------+
        | :attr:`~TimeUnit.BATCH`                             | - ``batch_size``                                | No required parameters.                                   | - ``batch_size``                                          | Unsupported conversion.           | - ``max_training_duration`` |
        |                                                     | - ``drop_last``                                 |                                                           | - ``drop_last``                                           |                                   |                             |
        |                                                     | - ``dataset_num_samples``                       |                                                           | - ``dataset_num_samples`` (if ``drop_last`` is ``False``) |                                   |                             |
        +-----------------------------------------------------+-------------------------------------------------+-----------------------------------------------------------+-----------------------------------------------------------+-----------------------------------+-----------------------------+
        | :attr:`~TimeUnit.SAMPLE`                            | - ``drop_last``                                 | - ``batch_size``                                          | No required parameters.                                   | Unsupported conversion.           | - ``max_training_duration`` |
        |                                                     | - ``dataset_num_samples``                       | - ``drop_last``                                           |                                                           |                                   |                             |
        |                                                     | - ``batch_size`` (if ``drop_last`` is ``True``) | - ``dataset_num_samples`` (if ``drop_last`` is ``False``) |                                                           |                                   |                             |
        +-----------------------------------------------------+-------------------------------------------------+-----------------------------------------------------------+-----------------------------------------------------------+-----------------------------------+-----------------------------+
        | :attr:`~TimeUnit.TOKEN`                             | - ``dataset_num_tokens``                        | Unsupported conversion.                                   | Unsupported conversion.                                   | No required parameters.           | - ``max_training_duration`` |
        |                                                     | - ``drop_last`` must be ``False``               |                                                           |                                                           |                                   |                             |
        +-----------------------------------------------------+-------------------------------------------------+-----------------------------------------------------------+-----------------------------------------------------------+-----------------------------------+-----------------------------+
        | :attr:`~TimeUnit.DURATION`                          | - ``max_training_duration``                     | - ``max_training_duration``                               | - ``max_training_duration``                               | - ``max_training_duration``       | No required parameters.     |
        +-----------------------------------------------------+-------------------------------------------------+-----------------------------------------------------------+-----------------------------------------------------------+-----------------------------------+-----------------------------+

        For example:

        >>> Time(2, "ep").convert(TimeUnit.BATCH, dataset_num_samples=100, batch_size=50, drop_last=True) == Time("4ba")

        Args:
            unit (Union[TimeUnit, str]): The desired unit to convert the time instance into.
            batch_size (int, optional): The optimization batch size.
            drop_last (bool, optional): Whether the dataloader is dropping last (incomplete) batches.
            dataset_num_samples (int, optional): The number of samples in the dataset.
            dataset_num_tokens (int, optional): The number of tokens in the dataset. Required only if
                converting to or from :attr:`TimeUnit.TOKEN`.
            max_training_duration (str or Time, optional): The total training duration. Required only
                if converting to or from :attr:`TimeUnit.DURATION`.
        
        Raises:
            ValueError: If it is not possible to perform the conversion. 

        Returns:
            Time: The time, in the specified ``unit``.
        """
        unit = TimeUnit(unit)

        if unit == self.unit:
            # No conversion required
            return Time(self.value, self.unit)

        if unit == TimeUnit.DURATION or self.unit == TimeUnit.DURATION:
            # if the desired unit is duration, then the logic is the same regardless of the from unit
            if max_training_duration is None:
                raise ValueError("max_training_duration is required to convert to or from DURATION")
            if unit == TimeUnit.DURATION:
                return self.convert_to_duration(max_training_duration=max_training_duration)
            else:
                return self.convert_from_duration(max_training_duration=max_training_duration)

        if self.unit == TimeUnit.EPOCH:
            if unit == TimeUnit.BATCH:
                if batch_size is None:
                    raise ValueError("batch_size is required to convert from EPOCH to BATCH")
                if drop_last is None:
                    raise ValueError("drop_last is required to convert from EPOCH to BATCH")
                if dataset_num_samples is None:
                    raise ValueError("dataset_num_samples is required to convert from EPOCH to BATCH")
                return self.convert_epoch_to_batch(batch_size=batch_size,
                                                   drop_last=drop_last,
                                                   dataset_num_samples=dataset_num_samples)
            if unit == TimeUnit.SAMPLE:
                if dataset_num_samples is None:
                    raise ValueError("dataset_num_samples is required to convert from EPOCH to SAMPLE")
                if drop_last is None:
                    raise ValueError("drop_last is required to convert from EPOCH to SAMPLE")
                return self.convert_epoch_to_sample(drop_last=drop_last,
                                                    dataset_num_samples=dataset_num_samples,
                                                    batch_size=batch_size)
            if unit == TimeUnit.TOKEN:
                if dataset_num_tokens is None:
                    raise ValueError("dataset_num_tokens is required to convert from EPOCH to TOKEN")
                return self.convert_epoch_to_token(dataset_num_tokens=dataset_num_tokens)
        if self.unit == TimeUnit.BATCH:
            if unit == TimeUnit.EPOCH:
                if batch_size is None:
                    raise ValueError("batch_size is required to convert from EPOCH to BATCH")
                if drop_last is None:
                    raise ValueError("drop_last is required to convert from EPOCH to BATCH")
                if dataset_num_samples is None:
                    raise ValueError("dataset_num_samples is required to convert from EPOCH to BATCH")
                return self.convert_batch_to_epoch(batch_size=batch_size,
                                                   drop_last=drop_last,
                                                   dataset_num_samples=dataset_num_samples)
            if unit == TimeUnit.SAMPLE:
                if batch_size is None:
                    raise ValueError("batch_size is required to convert from BATCH to SAMPLE")
                if drop_last is None:
                    raise ValueError("drop_last is required to convert from BATCH to SAMPLE")
                return self.convert_batch_to_sample(batch_size=batch_size,
                                                    drop_last=drop_last,
                                                    dataset_num_samples=dataset_num_samples)
        if self.unit == TimeUnit.SAMPLE:
            if unit == TimeUnit.EPOCH:
                if drop_last is None:
                    raise ValueError("drop_last is required to convert from SAMPLE to EPOCH")
                if dataset_num_samples is None:
                    raise ValueError("dataset_num_samples is required to convert from SAMPLE to SAMPLE")
                return self.convert_sample_to_epoch(drop_last=drop_last,
                                                    dataset_num_samples=dataset_num_samples,
                                                    batch_size=batch_size)
            if unit == TimeUnit.BATCH:
                if drop_last is None:
                    raise ValueError("drop_last is required to convert from BATCH to SAMPLE")
                if batch_size is None:
                    raise ValueError("batch_size is required to convert from BATCH to SAMPLE")
                return self.convert_sample_to_batch(batch_size=batch_size,
                                                    drop_last=drop_last,
                                                    dataset_num_samples=dataset_num_samples)
        if self.unit == TimeUnit.TOKEN:
            if unit == TimeUnit.EPOCH:
                if dataset_num_tokens is None:
                    raise ValueError("dataset_num_tokens is required to convert from TOKEN to EPOCH")
                return self.convert_token_to_epoch(dataset_num_tokens=dataset_num_tokens)

        raise ValueError(f"Unable to convert from {self.unit} to {unit}")

    def convert_epoch_to_batch(self, *, batch_size: int, drop_last: bool, dataset_num_samples: int) -> Time:
        """Convert ``self`` into :attr:`TimeUnit.BATCH`. Requires that ``self.unit == TimeUnit.EPOCH``.

        Args:
            batch_size (int): The optimization batch size.
            drop_last (bool): Whether the dataloader is dropping last (incomplete) batches.
            dataset_num_samples (int): The number of samples in the dataset.

        Raises:
            RuntimeError: Raised if ``self.unit != TimeUnit.EPOCH``

        Returns:
            Time: The time, in :attr:`TimeUnit.BATCH`.
        """
        if self.unit != TimeUnit.EPOCH:
            raise RuntimeError(f"Time {self} units are not epochs.")
        if drop_last:
            num_batches_per_epoch = dataset_num_samples // batch_size
        else:
            num_batches_per_epoch = (dataset_num_samples - 1) // batch_size + 1
        return Time(self.value * num_batches_per_epoch, TimeUnit.BATCH)

    def convert_epoch_to_sample(self,
                                *,
                                drop_last: bool,
                                dataset_num_samples: int,
                                batch_size: Optional[int] = None) -> Time:
        """Convert ``self`` into :attr:`TimeUnit.SAMPLE`. Requires that ``self.unit == TimeUnit.EPOCH``.

        Args:
            drop_last (bool): Whether the dataloader is dropping last (incomplete) batches.
            dataset_num_samples (int): The number of samples in the dataset.
            batch_size (int, optional): The optimization batch size. Required if ``drop_last`` is ``True``.

        Raises:
            RuntimeError: Raised if ``self.unit != TimeUnit.EPOCH``

        Returns:
            Time: The time, in :attr:`TimeUnit.SAMPLE`.
        """
        if self.unit != TimeUnit.EPOCH:
            raise RuntimeError(f"Time {self} units are not epochs.")
        if drop_last:
            if batch_size is None:
                raise ValueError("batch_size is required to convert from EPOCH to SAMPLE when drop_last=True")
            num_batches_per_epoch = dataset_num_samples // batch_size
            num_samples_per_epoch = num_batches_per_epoch * batch_size
            return Time(self.value * num_samples_per_epoch, TimeUnit.SAMPLE)
        else:
            return Time(self.value * dataset_num_samples, TimeUnit.SAMPLE)

    def convert_epoch_to_token(self, *, dataset_num_tokens: int) -> Time:
        """Convert ``self`` into :attr:`TimeUnit.TOKEN`. Requires that ``self.unit == TimeUnit.EPOCH``.

        .. note::

            The conversion is valid only if the dataloader yields all batches (i.e. ``drop_last`` is ``False``).

        Args:
            dataset_num_tokens (int): The number of tokens in the dataset.

        Raises:
            RuntimeError: Raised if ``self.unit != TimeUnit.EPOCH``

        Returns:
            Time: The time, in :attr:`TimeUnit.TOKEN`.
        """
        if self.unit != TimeUnit.EPOCH:
            raise RuntimeError(f"Time {self} units are not epochs.")
        return Time(self.value * dataset_num_tokens, TimeUnit.TOKEN)

    def convert_batch_to_epoch(self, *, batch_size: int, drop_last: bool, dataset_num_samples: int):
        """Convert ``self`` into :attr:`TimeUnit.EPOCH`. Requires that ``self.unit == TimeUnit.BATCH``.

        Args:
            batch_size (int): The optimization batch size.
            drop_last (bool): Whether the dataloader is dropping last (incomplete) batches.
            dataset_num_samples (int): The number of samples in the dataset.

        Raises:
            RuntimeError: Raised if ``self.unit != TimeUnit.BATCH``

        Returns:
            Time: The time, in :attr:`TimeUnit.EPOCH`.
        """
        if self.unit != TimeUnit.BATCH:
            raise RuntimeError(f"Time {self} units are not batches.")
        if drop_last:
            num_batches_per_epoch = dataset_num_samples // batch_size
        else:
            num_batches_per_epoch = (dataset_num_samples - 1) // batch_size + 1
        return Time(self.value // num_batches_per_epoch, TimeUnit.EPOCH)

    def convert_batch_to_sample(
        self,
        *,
        batch_size: int,
        drop_last: bool,
        dataset_num_samples: Optional[int] = None,
    ) -> Time:
        """Convert ``self`` into :attr:`TimeUnit.SAMPLE`. Requires that ``self.unit == TimeUnit.BATCH``.

        Args:
            batch_size (int): The optimization batch size.
            drop_last (bool): Whether the dataloader is dropping last (incomplete) batches.
            dataset_num_samples (int, optional): The number of samples in the dataset. Required if ``drop_last`` is ``False``.

        Raises:
            RuntimeError: Raised if ``self.unit != TimeUnit.BATCH``

        Returns:
            Time: The time, in :attr:`TimeUnit.SAMPLE`.
        """
        if self.unit != TimeUnit.BATCH:
            raise RuntimeError(f"Time {self} units are not batches.")
        if drop_last:
            return Time(self.value * batch_size, TimeUnit.SAMPLE)
        else:
            if dataset_num_samples is None:
                raise ValueError("dataset_num_samples is required to convert from EPOCH to BATCH when drop_last=False")
            num_batches_per_epoch = (dataset_num_samples - 1) // batch_size + 1
            samples_from_complete_epochs = (self.value // num_batches_per_epoch) * dataset_num_samples
            remaining_batches = self.value % num_batches_per_epoch
            samples_from_incomplete_epochs = remaining_batches * batch_size
            return Time(samples_from_complete_epochs + samples_from_incomplete_epochs, TimeUnit.SAMPLE)

    def convert_sample_to_epoch(self,
                                *,
                                drop_last: bool,
                                dataset_num_samples: int,
                                batch_size: Optional[int] = None) -> Time:
        """Convert ``self`` into :attr:`TimeUnit.EPOCH`. Requires that ``self.unit == TimeUnit.SAMPLE``.

        Args:
            drop_last (bool): Whether the dataloader is dropping last (incomplete) batches.
            dataset_num_samples (int): The number of samples in the dataset.
            batch_size (int, optional): The optimization batch size. Required if ``drop_last`` is ``True``.

        Raises:
            RuntimeError: Raised if ``self.unit != TimeUnit.SAMPLE``

        Returns:
            Time: The time, in :attr:`TimeUnit.EPOCH`.
        """
        if self.unit != TimeUnit.SAMPLE:
            raise RuntimeError(f"Time {self} units are not samples.")
        if drop_last:
            if batch_size is None:
                raise ValueError("batch_size is required to convert from SAMPLE to SAMPLE when drop_last=True")
            num_batches_per_epoch = dataset_num_samples // batch_size
            num_samples_per_epoch = num_batches_per_epoch * batch_size
            return Time(self.value // num_samples_per_epoch, TimeUnit.EPOCH)
        else:
            return Time(self.value // dataset_num_samples, TimeUnit.EPOCH)

    def convert_sample_to_batch(
        self,
        *,
        batch_size: int,
        drop_last: bool,
        dataset_num_samples: Optional[int] = None,
    ) -> Time:
        """Convert ``self`` into :attr:`TimeUnit.BATCH`. Requires that ``self.unit == TimeUnit.SAMPLE``.

        Args:
            batch_size (int): The optimization batch size.
            drop_last (bool): Whether the dataloader is dropping last (incomplete) batches.
            dataset_num_samples (int, optional): The number of samples in the dataset. Required if ``drop_last`` is ``False``.

        Raises:
            RuntimeError: Raised if ``self.unit != TimeUnit.SAMPLE``

        Returns:
            Time: The time, in :attr:`TimeUnit.BATCH`.
        """
        if self.unit != TimeUnit.SAMPLE:
            raise RuntimeError(f"Time {self} units are not samples.")
        if drop_last:
            return Time(self.value // batch_size, TimeUnit.BATCH)
        else:
            if dataset_num_samples is None:
                raise ValueError("dataset_num_samples is required to convert from SAMPLE to BATCH when drop_last=False")
            num_batches_per_epoch = (dataset_num_samples - 1) // batch_size + 1
            batches_from_complete_epochs = (self.value // dataset_num_samples) * num_batches_per_epoch
            batches_from_incomplete_epoch = (self.value % dataset_num_samples) // batch_size
            return Time(batches_from_complete_epochs + batches_from_incomplete_epoch, TimeUnit.BATCH)

    def convert_token_to_epoch(self, *, dataset_num_tokens: int) -> Time:
        """Convert ``self`` into :attr:`TimeUnit.EPOCH`. Requires that ``self.unit == TimeUnit.TOKEN``.

        .. note::

            The conversion is valid only if the dataloader yields all batches (i.e. ``drop_last`` == ``False``).

        Args:
            dataset_num_tokens (int): The number of tokens in the dataset.

        Raises:
            RuntimeError: Raised if ``self.unit != TimeUnit.TOKEN``

        Returns:
            Time: The time, in :attr:`TimeUnit.EPOCH`.
        """
        if self.unit != TimeUnit.TOKEN:
            raise RuntimeError(f"Time {self} units are not tokens.")
        return Time(self.value // dataset_num_tokens, TimeUnit.EPOCH)

    def convert_to_duration(self, *, max_training_duration: Union[str, Time]) -> Time:
        """Convert ``self`` into :attr:`TimeUnit.DURATION`.

        Args:
            max_training_duration (str or Time): The total training duration.

        Returns:
            Time: The time, in :attr:`TimeUnit.DURATION`.
        """
        if self.unit == TimeUnit.DURATION:
            return Time(self.value, self.unit)
        if isinstance(max_training_duration, str):
            max_training_duration = Time.from_timestring(max_training_duration)
        return self / max_training_duration

    def convert_from_duration(self, *, max_training_duration: Union[str, Time]) -> Time:
        """Convert ``self`` into the units of ``max_training_duration``.

        Args:
            max_training_duration (str or Time): The total training duration.

        Raises:
            RuntimeError: Raised if ``self.unit != TimeUnit.DURATION``

        Returns:
            Time: The time, in the units of ``max_training_duration``.
        """
        if self.unit != TimeUnit.DURATION:
            raise RuntimeError(f"Time {self} units is not duration.")
        if isinstance(max_training_duration, str):
            max_training_duration = Time.from_timestring(max_training_duration)
        if max_training_duration.unit == TimeUnit.DURATION:
            return Time(self.value, self.unit)
        return self * max_training_duration


class Timer(Serializable):
    """Timer tracks the current training progress, in terms of epochs, batches, samples, and tokens."""

    def __init__(self):
        self._epoch = Time(0, TimeUnit.EPOCH)
        self._batch = Time(0, TimeUnit.BATCH)
        self._sample = Time(0, TimeUnit.SAMPLE)
        self._token = Time(0, TimeUnit.TOKEN)

    def state_dict(self) -> StateDict:
        return {
            "epoch": self.epoch.value,
            "batch": self.batch.value,
            "sample": self.sample.value,
            "token": self.token.value,
        }

    def load_state_dict(self, state: StateDict) -> None:
        self._epoch = Time(state["epoch"], TimeUnit.EPOCH)
        self._batch = Time(state["batch"], TimeUnit.BATCH)
        self._sample = Time(state["sample"], TimeUnit.SAMPLE)
        self._token = Time(state["token"], TimeUnit.TOKEN)

    @property
    def epoch(self) -> Time:
        """The current epoch."""
        return self._epoch

    @property
    def batch(self) -> Time:
        """The current batch."""
        return self._batch

    @property
    def sample(self) -> Time:
        """The current sample."""
        return self._sample

    @property
    def token(self) -> Time:
        """The current token."""
        return self._token

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
        if isinstance(samples, int):
            samples = Time(samples, TimeUnit.SAMPLE)
        if isinstance(tokens, int):
            tokens = Time(tokens, TimeUnit.TOKEN)
        self._sample += samples
        self._token += tokens

    def on_epoch_complete(self):
        """Called by the trainer at the end of an epoch."""
        self._epoch += Time(1, TimeUnit.EPOCH)
