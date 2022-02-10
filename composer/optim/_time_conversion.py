# Copyright 2021 MosaicML. All Rights Reserved.

import textwrap
import warnings
from typing import Optional, Union

from composer.core import Time, TimeUnit


def convert(
    time: Union[str, Time],
    unit: Union[TimeUnit, str],
    *,
    steps_per_epoch: Optional[int] = None,
    samples_per_epoch: Optional[int] = None,
    dataset_num_tokens: Optional[int] = None,
    max_training_duration: Optional[Union[str, Time]] = None,
) -> Time:
    r"""Convert a :class:`Time` instance into the specified ``unit``.
    Parameter ``unit`` is always required. The following table lists the additional required parameters
    to perform the conversion:
    +-----------------------------------------------------+-----------------------------+------------------------------+-----------------------------+-----------------------------------+-----------------------------+
    | From Unit |:arrow_down:| \\ To Unit |:arrow_right:| | :attr:`~TimeUnit.EPOCH`     | :attr:`~TimeUnit.BATCH`      | :attr:`~TimeUnit.SAMPLE`    | :attr:`~TimeUnit.TOKEN`           | :attr:`~TimeUnit.DURATION`  |
    +-----------------------------------------------------+-----------------------------+------------------------------+-----------------------------+-----------------------------------+-----------------------------+
    | :attr:`~TimeUnit.EPOCH`                             | No required parameters.     | - ``steps_per_epoch``        | - ``samples_per_epoch``     | - ``dataset_num_tokens``          | - ``max_training_duration`` |
    +-----------------------------------------------------+-----------------------------+------------------------------+-----------------------------+-----------------------------------+-----------------------------+
    | :attr:`~TimeUnit.BATCH`                             | - ``steps_per_epoch``       | No required parameters.      | - ``steps_per_epoch``       | Unsupported conversion.           | - ``max_training_duration`` |
    |                                                     |                             |                              | - ``samples_per_epoch``     |                                   |                             |
    +-----------------------------------------------------+-----------------------------+------------------------------+-----------------------------+-----------------------------------+-----------------------------+
    | :attr:`~TimeUnit.SAMPLE`                            | - ``samples_per_epoch``     | - ``steps_per_epoch``        | No required parameters.     | Unsupported conversion.           | - ``max_training_duration`` |
    |                                                     |                             | - ``samples_per_epoch``      |                             |                                   |                             |
    +-----------------------------------------------------+-----------------------------+------------------------------+-----------------------------+-----------------------------------+-----------------------------+
    | :attr:`~TimeUnit.TOKEN`                             | - ``dataset_num_tokens``    | Unsupported conversion.      | Unsupported conversion.     | No required parameters.           | - ``max_training_duration`` |
    +-----------------------------------------------------+-----------------------------+------------------------------+-----------------------------+-----------------------------------+-----------------------------+
    | :attr:`~TimeUnit.DURATION`                          | - ``max_training_duration`` | - ``max_training_duration``  | - ``max_training_duration`` | - ``max_training_duration``       | No required parameters.     |
    +-----------------------------------------------------+-----------------------------+------------------------------+-----------------------------+-----------------------------------+-----------------------------+
    Args:
        unit (Union[TimeUnit, str]): The desired unit to convert the time instance into.
        steps_per_epoch (int, optional): The number of optimization steps per epoch.
        samples_per_epoch (int, optional): The number of samples per epoch.
        dataset_num_tokens (int, optional): The number of tokens in the dataset. Required only if
            converting to or from :attr:`TimeUnit.TOKEN`.
        max_training_duration (str or Time, optional): The total training duration. Required only
            if converting to or from :attr:`TimeUnit.DURATION`.

    Raises:
        ValueError: If it is not possible to perform the conversion.

    Returns:
        Time: The time, in the specified ``unit``.
    """
    warnings.warn(textwrap.dedent("""\
        TimeDeprecationWarning: Time conversion is deprecated.
        Instead, please use closed-loop calculations that depend on the current training progress
        (available via state.timer) and the total training duration (available via state.max_duration)"""),
                  category=DeprecationWarning)
    if isinstance(time, str):
        time = Time.from_timestring(time)
    unit = TimeUnit(unit)

    if unit == time.unit:
        # No conversion required
        return Time(time.value, time.unit)

    if unit == TimeUnit.DURATION or time.unit == TimeUnit.DURATION:
        # if the desired unit is duration, then the logic is the same regardless of the from unit
        if max_training_duration is None:
            raise ValueError("max_training_duration is required to convert to or from DURATION")
        if isinstance(max_training_duration, str):
            max_training_duration = Time.from_timestring(max_training_duration)
        max_training_duration_unit = max_training_duration.unit
        if unit == TimeUnit.DURATION:
            time_in_max_duration_unit = convert(time,
                                                max_training_duration_unit,
                                                steps_per_epoch=steps_per_epoch,
                                                samples_per_epoch=samples_per_epoch,
                                                dataset_num_tokens=dataset_num_tokens)
            return _convert_to_duration(time_in_max_duration_unit, max_training_duration=max_training_duration)
        else:
            max_training_duration_in_units = convert(max_training_duration,
                                                     unit,
                                                     steps_per_epoch=steps_per_epoch,
                                                     samples_per_epoch=samples_per_epoch,
                                                     dataset_num_tokens=dataset_num_tokens)
            converted_time = _convert_from_duration(time, max_training_duration=max_training_duration_in_units)
            return converted_time

    if time.unit == TimeUnit.EPOCH:
        if unit == TimeUnit.BATCH:
            if steps_per_epoch is None:
                raise ValueError("steps_per_epoch is required to convert from EPOCH to BATCH")
            return _convert_epoch_to_batch(time, steps_per_epoch=steps_per_epoch)
        if unit == TimeUnit.SAMPLE:
            if samples_per_epoch is None:
                raise ValueError("samples_per_epoch is required to convert from EPOCH to SAMPLE")
            return _convert_epoch_to_sample(time, samples_per_epoch=samples_per_epoch)
        if unit == TimeUnit.TOKEN:
            if dataset_num_tokens is None:
                raise ValueError("dataset_num_tokens is required to convert from EPOCH to TOKEN")
            return _convert_epoch_to_token(time, dataset_num_tokens=dataset_num_tokens)
    if time.unit == TimeUnit.BATCH:
        if unit == TimeUnit.EPOCH:
            if steps_per_epoch is None:
                raise ValueError("samples_per_epoch is required to convert from EPOCH to BATCH")
            return _convert_batch_to_epoch(time, steps_per_epoch=steps_per_epoch)
        if unit == TimeUnit.SAMPLE:
            if steps_per_epoch is None:
                raise ValueError("steps_per_epoch is required to convert from BATCH to SAMPLE")
            if samples_per_epoch is None:
                raise ValueError("samples_per_epoch is required to convert from BATCH to SAMPLE")
            return _convert_batch_to_sample(time, steps_per_epoch=steps_per_epoch, samples_per_epoch=samples_per_epoch)
    if time.unit == TimeUnit.SAMPLE:
        if unit == TimeUnit.EPOCH:
            if samples_per_epoch is None:
                raise ValueError("samples_per_epoch is required to convert from SAMPLE to SAMPLE")
            return _convert_sample_to_epoch(time, samples_per_epoch=samples_per_epoch)
        if unit == TimeUnit.BATCH:
            if samples_per_epoch is None:
                raise ValueError("samples_per_epoch is required to convert from BATCH to SAMPLE")
            if steps_per_epoch is None:
                raise ValueError("steps_per_epoch is required to convert from BATCH to SAMPLE")
            return _convert_sample_to_batch(time, steps_per_epoch=steps_per_epoch, samples_per_epoch=samples_per_epoch)
    if time.unit == TimeUnit.TOKEN:
        if unit == TimeUnit.EPOCH:
            if dataset_num_tokens is None:
                raise ValueError("dataset_num_tokens is required to convert from TOKEN to EPOCH")
            return _convert_token_to_epoch(time, dataset_num_tokens=dataset_num_tokens)

    raise ValueError(f"Unable to convert from {time.unit} to {unit}")


def _convert_epoch_to_batch(time: Time[int], *, steps_per_epoch: int) -> Time[int]:
    """Convert ``time`` into :attr:`TimeUnit.BATCH`. Requires that ``time.unit == TimeUnit.EPOCH``.

    Args:
        time (Time): The time
        steps_per_epoch (int): The number of optimizations steps per epoch.

    Raises:
        RuntimeError: Raised if ``time.unit != TimeUnit.EPOCH``

    Returns:
        Time: The time, in :attr:`TimeUnit.BATCH`.
    """
    if time.unit != TimeUnit.EPOCH:
        raise RuntimeError(f"Time {time} units are not epochs.")
    return Time(time.value * steps_per_epoch, TimeUnit.BATCH)


def _convert_epoch_to_sample(time: Time[int], *, samples_per_epoch: int) -> Time[int]:
    """Convert ``time`` into :attr:`TimeUnit.SAMPLE`. Requires that ``time.unit == TimeUnit.EPOCH``.

    Args:
        time (Time): The time
        samples_per_epoch (int): The number of samples trained per epoch.

    Raises:
        RuntimeError: Raised if ``time.unit != TimeUnit.EPOCH``

    Returns:
        Time: The time, in :attr:`TimeUnit.SAMPLE`.
    """
    if time.unit != TimeUnit.EPOCH:
        raise RuntimeError(f"Time {time} units are not epochs.")

    return Time(time.value * samples_per_epoch, TimeUnit.SAMPLE)


def _convert_epoch_to_token(time: Time[int], *, dataset_num_tokens: int) -> Time[int]:
    """Convert ``time`` into :attr:`TimeUnit.TOKEN`. Requires that ``time.unit == TimeUnit.EPOCH``.

    .. note::

        The conversion is valid only if the dataloader yields all batches (i.e. ``drop_last`` is ``False``).

    Args:
        time (Time): The time
        dataset_num_tokens (int): The number of tokens in the dataset.

    Raises:
        RuntimeError: Raised if ``time.unit != TimeUnit.EPOCH``

    Returns:
        Time: The time, in :attr:`TimeUnit.TOKEN`.
    """
    if time.unit != TimeUnit.EPOCH:
        raise RuntimeError(f"Time {time} units are not epochs.")
    return Time(time.value * dataset_num_tokens, TimeUnit.TOKEN)


def _convert_batch_to_epoch(time: Time[int], *, steps_per_epoch: int) -> Time[int]:
    """Convert ``time`` into :attr:`TimeUnit.EPOCH`. Requires that ``time.unit == TimeUnit.BATCH``.

    Args:
        time (Time): The time
        steps_per_epoch (int): The optimization batch size.

    Raises:
        RuntimeError: Raised if ``time.unit != TimeUnit.BATCH``
    Returns:
        Time: The time, in :attr:`TimeUnit.EPOCH`.
    """
    if time.unit != TimeUnit.BATCH:
        raise RuntimeError(f"Time {time} units are not batches.")
    return Time(time.value // steps_per_epoch, TimeUnit.EPOCH)


def _convert_batch_to_sample(
    time: Time[int],
    *,
    steps_per_epoch: int,
    samples_per_epoch: int,
) -> Time[int]:
    """Convert ``time`` into :attr:`TimeUnit.SAMPLE`. Requires that ``time.unit == TimeUnit.BATCH``.

    Args:
        time (Time): The time
        steps_per_epoch (int): The number of optimization steps per epoch.
        samples_per_epoch (int): The number of samples per epoch.

    Raises:
        RuntimeError: Raised if ``time.unit != TimeUnit.BATCH``

    Returns:
        Time: The time, in :attr:`TimeUnit.SAMPLE`.
    """
    if time.unit != TimeUnit.BATCH:
        raise RuntimeError(f"Time {time} units are not batches.")
    if samples_per_epoch % steps_per_epoch != 0:
        raise ValueError("Cannot determine the batch size as samples_per_epoch %% steps_per_epoch != 0")
    batch_size = samples_per_epoch // steps_per_epoch

    return Time(time.value * batch_size, TimeUnit.SAMPLE)


def _convert_sample_to_epoch(time: Time[int], *, samples_per_epoch: int) -> Time[int]:
    """Convert ``time`` into :attr:`TimeUnit.EPOCH`. Requires that ``time.unit == TimeUnit.SAMPLE``.

    Args:
        time (Time): The time
        samples_per_epoch (int): The number of samples per epoch.

    Raises:
        RuntimeError: Raised if ``time.unit != TimeUnit.SAMPLE``
    Returns:
        Time: The time, in :attr:`TimeUnit.EPOCH`.
    """
    if time.unit != TimeUnit.SAMPLE:
        raise RuntimeError(f"Time {time} units are not samples.")
    return Time(time.value // samples_per_epoch, TimeUnit.EPOCH)


def _convert_sample_to_batch(
    time: Time[int],
    *,
    steps_per_epoch: int,
    samples_per_epoch: int,
) -> Time[int]:
    """Convert ``time`` into :attr:`TimeUnit.BATCH`. Requires that ``time.unit == TimeUnit.SAMPLE``.

    Args:
        time (Time): The time
        steps_per_epoch (int): The number of optimization steps per epoch.
        samples_per_epoch (int): The number of samples per epoch.

    Raises:
        RuntimeError: Raised if ``time.unit != TimeUnit.SAMPLE``

    Returns:
        Time: The time, in :attr:`TimeUnit.BATCH`.
    """
    if time.unit != TimeUnit.SAMPLE:
        raise RuntimeError(f"Time {time} units are not samples.")

    if samples_per_epoch % steps_per_epoch != 0:
        raise ValueError("Cannot determine the batch size as samples_per_epoch %% steps_per_epoch != 0")
    batch_size = samples_per_epoch // steps_per_epoch
    return Time(time.value // batch_size, TimeUnit.BATCH)


def _convert_token_to_epoch(time: Time[int], *, dataset_num_tokens: int) -> Time[int]:
    """Convert ``time`` into :attr:`TimeUnit.EPOCH`. Requires that ``time.unit == TimeUnit.TOKEN``.

    .. note::

        The conversion is valid only if the dataloader yields all batches (i.e. ``drop_last`` == ``False``).

    Args:
        time (Time): The time
        dataset_num_tokens (int): The number of tokens in the dataset.

    Raises:
        RuntimeError: Raised if ``time.unit != TimeUnit.TOKEN``

    Returns:
        Time: The time, in :attr:`TimeUnit.EPOCH`.
    """
    if time.unit != TimeUnit.TOKEN:
        raise RuntimeError(f"Time {time} units are not tokens.")
    return Time(time.value // dataset_num_tokens, TimeUnit.EPOCH)


def _convert_to_duration(time: Time, *, max_training_duration: Union[str, Time[int]]) -> Time[float]:
    """Convert ``time`` into :attr:`TimeUnit.DURATION`.

    Args:
        time (Time): The time
        max_training_duration (str or Time): The total training duration.

    Returns:
        Time: The time, in :attr:`TimeUnit.DURATION`.
    """
    if time.unit == TimeUnit.DURATION:
        return Time(time.value, time.unit)
    if isinstance(max_training_duration, str):
        max_training_duration = Time.from_timestring(max_training_duration)
    return time / max_training_duration


def _convert_from_duration(time: Time[float], *, max_training_duration: Union[str, Time[int]]) -> Time:
    """Convert ``time`` into the units of ``max_training_duration``.

    Args:
        time (Time): The time
        max_training_duration (str or Time): The total training duration.

    Raises:
        RuntimeError: Raised if ``time.unit != TimeUnit.DURATION``

    Returns:
        Time: The time, in the units of ``max_training_duration``.
    """
    if time.unit != TimeUnit.DURATION:
        raise RuntimeError(f"Time {time} units is not duration.")
    if isinstance(max_training_duration, str):
        max_training_duration = Time.from_timestring(max_training_duration)
    if max_training_duration.unit == TimeUnit.DURATION:
        return Time(time.value, time.unit)
    return time * max_training_duration
