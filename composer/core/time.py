# Copyright 2021 MosaicML. All Rights Reserved.

from __future__ import annotations

import re
import textwrap
import warnings
from typing import Optional, Tuple, Union

from composer.core.serializable import Serializable
from composer.core.types import StateDict
from composer.utils.string_enum import StringEnum


class TimeUnit(StringEnum):
    """Units of time for the training process.

    Attributes:
        EPOCH (str): Epochs.
        BATCH (str): Batchs (i.e. number of optimization steps)
        SAMPLE (str): Samples.
        TOKEN (str): Tokens. Applicable for natural language processing (NLP) models.
        DURATION (str): Fraction of the training process complete, on ``[0.0; 1.0]``
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
    """Time describes the progress of a training run, in terms of a :class:`TimeUnit` (epochs, batches,
    samples, tokens, or duration).

    To construct an instance of :class:`Time`, you can either:

        #. Use a time string. A time string is a numerical value followed by the value of a
            :class:`TimeUnit` enum. For example:

            >>> Time("5ep")  # describes 5 epochs.
            >>> Time("3e4tok")  # describes 30,000 tokens.
            >>> Time("0.5dur")  # describes 50% of the training process.
        
        #. Use a value followed by a :class:`TimeUnit` enum or string. For example,

            >>> Time(5, TimeUnit.EPOCH)  # describes 5 epochs.
            >>> Time(3e4, "tok")  # describes 30,000 tokens.
            >>> Time(0.5, "dur")  # describes 50% of the training process.
        
        #. Use a keyword argument for the unit. For example,

            >>> Time(epoch=5)  # describes 5 epochs.
            >>> Time(token=3e4)  # describes 30,000 tokens.
            >>> Time(dur=0.5)  # describes 50% of the training process.

    :class:`Time` supports addition and subtraction with other :class:`Time` instances that share the same
    :class:`TimeUnit`. For example:
    >>> Time(epoch=1) + Time(epoch=2) == Time("3ep")

    :class:`Time` supports multiplication when the multiplier or multiplicand is in units :attr:`TimeUnit.DURATION`.
    For example:
    >>> Time(epoch=2) * Time(0.5, "dur) == Time("1ep")

    :class:`Time` also supports division when both units are the same type. The result is in :attr:`TimeUnit.DURATION`.
    For example:
    >>> Time(epoch=2) / Time(epoch=4) == Time("0.5dur")



    Args:
        time (str, Time, int, or float, optional): If spcified, a time string, existing :class:`Time` instance,
            or number.
            If a time string or existing instance, then all other arguments should be left blank.
            If a number, then ``unit`` must also be specifed.
        unit (str | TimeUnit, optional): If ``time`` is number, the :class:`TimeUnit` corresponding to it.
        epoch (int, optional): Number of epochs. If specified, all other arguments should be left blank.
        batch (int, optional): Number of batches. If specified, all other arguments should be left blank.
        sample (int, optional): Number of samples. If specified, all other arguments should be left blank.
        token (int, optional): Number of tokens. If specified, all other arguments should be left blank.
        duration (float, optional): Fraction of the total training duration, on ``[0;1]``.
            If specified, all other arguments should be left blank.
    """

    def __init__(
        self,
        time: Optional[Union[str, Time, int, float]] = None,
        unit: Optional[Union[str, TimeUnit]] = None,
        *,
        epoch: Optional[int] = None,
        batch: Optional[int] = None,
        sample: Optional[int] = None,
        token: Optional[int] = None,
        duration: float = None,
    ):

        if (time is not None) + (epoch is not None) + (batch is not None) + (sample is not None) + (
                token is not None) + (duration is not None) != 1:
            raise ValueError("Exactly one argument should be speciifed")

        if time is not None:
            if isinstance(time, str):
                self._value, self._unit = self._parse_time_string(time)
            elif isinstance(time, Time):
                self._value, self._unit = time.value, time.unit
            else:
                if unit is None:
                    raise ValueError("when time is an integer, then unit must be specified")
                self._value, self._unit = time, TimeUnit(unit)

        elif epoch is not None:
            self._value = epoch
            self._unit = TimeUnit.EPOCH

        elif batch is not None:
            self._value = batch
            self._unit = TimeUnit.BATCH

        elif sample is not None:
            self._value = sample
            self._unit = TimeUnit.SAMPLE

        elif token is not None:
            self._value = token
            self._unit = TimeUnit.TOKEN

        elif duration is not None:
            self._value = duration
            self._unit = TimeUnit.DURATION
        else:
            raise ValueError("No value specified")

    @property
    def value(self) -> Union[int, float]:
        """The value of the time, as a number."""
        return self._value

    @property
    def unit(self) -> TimeUnit:
        """The unit of the time."""
        return self._unit

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.unit.name.lower()}={self.value})"

    def _convert(self, other: object) -> Time:
        if isinstance(other, Time):
            return other

        if isinstance(other, (int, float)):
            other_converted = Time(f"{other}{self.unit.value}")
            warnings.warn(
                textwrap.dedent(f"""TimeImplicitIntegerConversion:
                Implciitely converting {other} to {other_converted}. Assuming that the unit is {self.unit.name}.
                To eliminate this warning, replace {other} with {other_converted}."""))
            return other_converted

        if isinstance(other, str):
            other_converted = Time(other)
            warnings.warn(
                textwrap.dedent(f"""TimeImplicitStringConversion:
                Implciitely converting {other} to {other_converted}.
                To fix this warning, replace {other} with {other_converted}."""))
            return other_converted

        raise NotImplementedError(f"Cannot convert type {other} to {self.__class__.__name__}")

    def _cmp(self, other: object) -> int:
        other = self._convert(other)
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
        other = self._convert(other)
        if self.unit != other.unit:
            raise RuntimeError(f"Cannot add {self} to {other} since they have different units.")
        return Time(self.value + other.value, self.unit)

    def __radd__(self, other: object):
        return self + other

    def __sub__(self, other: object):
        other = self._convert(other)
        if self.unit != other.unit:
            raise RuntimeError(f"Cannot subtract {other} from {self} since they have different units.")
        return Time(self.value - other.value, self.unit)

    def __rsub__(self, other: object):
        return (-self) + other

    def __neg__(self):
        return Time(-self.value, self.unit)

    def __pos__(self):
        return Time(self)

    def __int__(self):
        return int(self.value)

    def __float__(self):
        return float(self.value)

    def __truediv__(self, other: object):
        other = self._convert(other)
        if self.unit != other.unit:
            raise RuntimeError(f"Cannot divide {self} by {other} since they have different units.")
        return Time(self.value / other.value, TimeUnit.DURATION)

    def __mul__(self, other: object):
        other = self._convert(other)
        if other.unit != TimeUnit.DURATION and self.unit != TimeUnit.DURATION:
            raise RuntimeError(f"Multiplication is supported only if one of the units is Duration")
        real_unit = self.unit if other.unit == TimeUnit.DURATION else other.unit
        return Time(self.value * other.value, real_unit)

    def __rmul__(self, other: object):
        return self * other

    @staticmethod
    def _parse_time_string(timestring: str) -> Tuple[Union[int, float], TimeUnit]:
        # Parses a time string into a (value, TimeUnit) tuple
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
        return value, unit

    def convert(
        self,
        unit: Union[TimeUnit, str],
        *,
        dataset_num_samples: Optional[int] = None,
        dataset_num_tokens: Optional[int] = None,
        max_training_duration: Optional[Union[str, Time]] = None,
        batch_size: Optional[int] = None,
        drop_last: Optional[bool] = None,
    ) -> Time:
        """Convert a :class:`Time` instance into the specified ``unit``.

        Only ``unit`` is always required. Other parameters may be required, depending on the conversion
        being performed.

        For example:
        >>> Time("2ep").convert(TimeUnit.BATCH, dataset_num_samples=100, batch_size=50, drop_last=True) == Time("4ba")

        Args:
            unit (Union[TimeUnit, str]): The desired unit to convert the time instance into.
            dataset_num_samples (int, optional): The number of samples in the dataset.
            dataset_num_tokens (int, optional): The number of tokens in the dataset. Required only if
                converting to or from :attr;`TimeUnit.TOKEN`.
            max_training_duration (str or Time, optional): The total training duration. Required only
                if converting to or from :attr:`TimeUnit.DURATION`.
            batch_size (int, optional): The optimization batch size.
            drop_last (bool, optional): Whether the dataloader is dropping last (incomplete) batches.
        
        Raises:
            ValueError: If it is not possible to perform the conversion. 

        Returns:
            Time: The time, in the specified ``unit``.
        """
        unit = TimeUnit(unit)

        if unit == TimeUnit.DURATION or self.unit == TimeUnit.DURATION:
            # if the desired unit is duration, then the logic is the same regardless of the from unit
            if self.unit == TimeUnit.DURATION and unit == TimeUnit.DURATION:
                return Time(self)
            if max_training_duration is None:
                raise ValueError("max_training_duration is required to convert to or from DURATION")
            max_training_duration = Time(max_training_duration)
            if unit == TimeUnit.DURATION:
                # converting to duration
                return self / max_training_duration
            else:
                # converting from druation
                return self * max_training_duration

        if self.unit == TimeUnit.EPOCH:
            if unit == TimeUnit.EPOCH:
                return Time(self)
            if unit == TimeUnit.BATCH:
                if batch_size is None:
                    raise ValueError("batch_size is required to convert from EPOCH to BATCH")
                if drop_last is None:
                    raise ValueError("drop_last is required to convert from EPOCH to BATCH")
                if dataset_num_samples is None:
                    raise ValueError("dataset_num_samples is required to convert from EPOCH to BATCH")
                if drop_last:
                    num_batches_per_epoch = dataset_num_samples // batch_size
                else:
                    num_batches_per_epoch = (dataset_num_samples - 1) // batch_size + 1
                return Time(self.value * num_batches_per_epoch, TimeUnit.BATCH)
            if unit == TimeUnit.SAMPLE:
                if dataset_num_samples is None:
                    raise ValueError("dataset_num_samples is required to convert from EPOCH to SAMPLE")
                if drop_last is None:
                    raise ValueError("drop_last is required to convert from EPOCH to SAMPLE")
                if drop_last:
                    if batch_size is None:
                        raise ValueError("batch_size is required to convert from EPOCH to SAMPLE when drop_last=True")
                    num_batches_per_epoch = dataset_num_samples // batch_size
                    num_samples_per_epoch = num_batches_per_epoch * batch_size
                    return Time(self.value * num_samples_per_epoch, TimeUnit.SAMPLE)
                else:
                    return Time(self.value * dataset_num_samples, TimeUnit.SAMPLE)
            if unit == TimeUnit.TOKEN:
                if dataset_num_tokens is None:
                    raise ValueError("dataset_num_tokens is required to convert from EPOCH to TOKEN")
                if drop_last is None:
                    raise ValueError("drop_last is required to convert from EPOCH to TOKEN")
                if drop_last is True:
                    raise ValueError("Cannot convert from EPOCH to TOKEN when drop_last=True")
                return Time(self.value * dataset_num_tokens, TimeUnit.TOKEN)
            raise ValueError(f"invalid unit: {unit}")
        if self.unit == TimeUnit.BATCH:
            if unit == TimeUnit.EPOCH:
                if batch_size is None:
                    raise ValueError("batch_size is required to convert from EPOCH to BATCH")
                if drop_last is None:
                    raise ValueError("drop_last is required to convert from EPOCH to BATCH")
                if dataset_num_samples is None:
                    raise ValueError("dataset_num_samples is required to convert from EPOCH to BATCH")
                if drop_last:
                    num_batches_per_epoch = dataset_num_samples // batch_size
                else:
                    num_batches_per_epoch = (dataset_num_samples - 1) // batch_size + 1
                return Time(self.value // num_batches_per_epoch, TimeUnit.EPOCH)
            if unit == TimeUnit.BATCH:
                return Time(self)
            if unit == TimeUnit.SAMPLE:
                if batch_size is None:
                    raise ValueError("batch_size is required to convert from BATCH to SAMPLE")
                if drop_last is None:
                    raise ValueError("drop_last is required to convert from BATCH to SAMPLE")
                if drop_last:
                    return Time(self.value * batch_size, TimeUnit.SAMPLE)
                else:
                    if dataset_num_samples is None:
                        raise ValueError(
                            "dataset_num_samples is required to convert from EPOCH to BATCH when drop_last=False")
                    num_batches_per_epoch = (dataset_num_samples - 1) // batch_size + 1
                    samples_from_complete_epochs = (self.value // num_batches_per_epoch) * dataset_num_samples
                    remaining_batches = self.value % num_batches_per_epoch
                    samples_from_incomplete_epochs = remaining_batches * batch_size
                    return Time(samples_from_complete_epochs + samples_from_incomplete_epochs, TimeUnit.SAMPLE)
            if unit == TimeUnit.TOKEN:
                raise ValueError("Cannot convert from BATCH to TOKEN")
            raise ValueError(f"invalid unit: {unit}")
        if self.unit == TimeUnit.SAMPLE:
            if unit == TimeUnit.EPOCH:
                if drop_last is None:
                    raise ValueError("drop_last is required to convert from SAMPLE to EPOCH")
                if dataset_num_samples is None:
                    raise ValueError("dataset_num_samples is required to convert from SAMPLE to SAMPLE")
                if drop_last:
                    if batch_size is None:
                        raise ValueError("batch_size is required to convert from SAMPLE to SAMPLE when drop_last=True")
                    num_batches_per_epoch = dataset_num_samples // batch_size
                    num_samples_per_epoch = num_batches_per_epoch * batch_size
                    return Time(self.value // num_samples_per_epoch, TimeUnit.EPOCH)
                else:
                    return Time(self.value // dataset_num_samples, TimeUnit.EPOCH)
            if unit == TimeUnit.BATCH:
                if drop_last is None:
                    raise ValueError("drop_last is required to convert from BATCH to SAMPLE")
                if batch_size is None:
                    raise ValueError("batch_size is required to convert from BATCH to SAMPLE")
                if drop_last:
                    return Time(self.value // batch_size, TimeUnit.BATCH)
                else:
                    if dataset_num_samples is None:
                        raise ValueError(
                            "dataset_num_samples is required to convert from SAMPLE to BATCH when drop_last=False")
                    num_batches_per_epoch = (dataset_num_samples - 1) // batch_size + 1
                    num_samples_per_epoch = num_batches_per_epoch * batch_size
                    batches_from_complete_epochs = (self.value // dataset_num_samples) * num_batches_per_epoch
                    batches_from_incomplete_epoch = (self.value % dataset_num_samples) // batch_size
                    return Time(batches_from_complete_epochs + batches_from_incomplete_epoch, TimeUnit.BATCH)
            if unit == TimeUnit.SAMPLE:
                return Time(self)
            if unit == TimeUnit.TOKEN:
                raise ValueError("Cannot convert from SAMPLE to TOKEN")
            raise ValueError(f"invalid unit: {unit}")
        if self.unit == TimeUnit.TOKEN:
            if unit == TimeUnit.EPOCH:
                if dataset_num_tokens is None:
                    raise ValueError("dataset_num_tokens is required to convert from TOKEN to EPOCH")
                if drop_last is None:
                    raise ValueError("drop_last is required to convert from TOKEN to EPOCH")
                if drop_last is True:
                    raise ValueError("Cannot convert from TOKEN to EPOCH when drop_last=True")
                return Time(self.value // dataset_num_tokens, TimeUnit.EPOCH)
            if unit == TimeUnit.BATCH:
                raise ValueError("Cannot convert from TOKEN to BATCH")
            if unit == TimeUnit.SAMPLE:
                raise ValueError("Cannot convert from TOKEN to SAMPLE")
            if unit == TimeUnit.TOKEN:
                return Time(self)
            raise ValueError(f"invalid unit: {unit}")
        raise RuntimeError("invalid from unit")


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
    def epoch(self):
        """The current epoch."""
        return self._epoch

    @property
    def batch(self):
        """The current batch."""
        return self._batch

    @property
    def sample(self):
        """The current sample."""
        return self._sample

    @property
    def token(self):
        """The current token."""
        return self._token

    def on_batch_complete(self, samples: Union[int, Time] = 0, tokens: Union[int, Time] = 0):
        """Called by the trainer at the end of every optimization batch.

        .. note::

            For accurate time tracking, the trainer is responsible for accumulating the total number of
            samples and/or tokens trained across all ranks before invoking this function. 

        Args:
            samples (int or Time, optional). The number of samples trained in the batch. Defaults to 0.
            tokens (int or Time, optional): The number of tokens trained in the batch. Defaults to 0.
        """
        self._batch += 1
        self._sample += Time(samples, TimeUnit.SAMPLE)
        self._token += Time(tokens, TimeUnit.TOKEN)

    def on_epoch_complete(self):
        """Called by the trainer at the end of an epoch."""
        self._epoch += 1
