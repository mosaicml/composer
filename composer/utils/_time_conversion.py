# Copyright 2021 MosaicML. All Rights Reserved.

from typing import Optional, Union

from composer.core import Time, TimeUnit


def convert(
    time: Union[str, Time],
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
        if unit == TimeUnit.DURATION:
            return convert_to_duration(time, max_training_duration=max_training_duration)
        else:
            return convert_from_duration(time, max_training_duration=max_training_duration)

    if time.unit == TimeUnit.EPOCH:
        if unit == TimeUnit.BATCH:
            if batch_size is None:
                raise ValueError("batch_size is required to convert from EPOCH to BATCH")
            if drop_last is None:
                raise ValueError("drop_last is required to convert from EPOCH to BATCH")
            if dataset_num_samples is None:
                raise ValueError("dataset_num_samples is required to convert from EPOCH to BATCH")
            return convert_epoch_to_batch(time,
                                          batch_size=batch_size,
                                          drop_last=drop_last,
                                          dataset_num_samples=dataset_num_samples)
        if unit == TimeUnit.SAMPLE:
            if dataset_num_samples is None:
                raise ValueError("dataset_num_samples is required to convert from EPOCH to SAMPLE")
            if drop_last is None:
                raise ValueError("drop_last is required to convert from EPOCH to SAMPLE")
            return convert_epoch_to_sample(time,
                                           drop_last=drop_last,
                                           dataset_num_samples=dataset_num_samples,
                                           batch_size=batch_size)
        if unit == TimeUnit.TOKEN:
            if dataset_num_tokens is None:
                raise ValueError("dataset_num_tokens is required to convert from EPOCH to TOKEN")
            return convert_epoch_to_token(time, dataset_num_tokens=dataset_num_tokens)
    if time.unit == TimeUnit.BATCH:
        if unit == TimeUnit.EPOCH:
            if batch_size is None:
                raise ValueError("batch_size is required to convert from EPOCH to BATCH")
            if drop_last is None:
                raise ValueError("drop_last is required to convert from EPOCH to BATCH")
            if dataset_num_samples is None:
                raise ValueError("dataset_num_samples is required to convert from EPOCH to BATCH")
            return convert_batch_to_epoch(time,
                                          batch_size=batch_size,
                                          drop_last=drop_last,
                                          dataset_num_samples=dataset_num_samples)
        if unit == TimeUnit.SAMPLE:
            if batch_size is None:
                raise ValueError("batch_size is required to convert from BATCH to SAMPLE")
            if drop_last is None:
                raise ValueError("drop_last is required to convert from BATCH to SAMPLE")
            return convert_batch_to_sample(time,
                                           batch_size=batch_size,
                                           drop_last=drop_last,
                                           dataset_num_samples=dataset_num_samples)
    if time.unit == TimeUnit.SAMPLE:
        if unit == TimeUnit.EPOCH:
            if drop_last is None:
                raise ValueError("drop_last is required to convert from SAMPLE to EPOCH")
            if dataset_num_samples is None:
                raise ValueError("dataset_num_samples is required to convert from SAMPLE to SAMPLE")
            return convert_sample_to_epoch(time,
                                           drop_last=drop_last,
                                           dataset_num_samples=dataset_num_samples,
                                           batch_size=batch_size)
        if unit == TimeUnit.BATCH:
            if drop_last is None:
                raise ValueError("drop_last is required to convert from BATCH to SAMPLE")
            if batch_size is None:
                raise ValueError("batch_size is required to convert from BATCH to SAMPLE")
            return convert_sample_to_batch(time,
                                           batch_size=batch_size,
                                           drop_last=drop_last,
                                           dataset_num_samples=dataset_num_samples)
    if time.unit == TimeUnit.TOKEN:
        if unit == TimeUnit.EPOCH:
            if dataset_num_tokens is None:
                raise ValueError("dataset_num_tokens is required to convert from TOKEN to EPOCH")
            return convert_token_to_epoch(time, dataset_num_tokens=dataset_num_tokens)

    raise ValueError(f"Unable to convert from {time.unit} to {unit}")


def convert_epoch_to_batch(time, *, batch_size: int, drop_last: bool, dataset_num_samples: int) -> Time:
    """Convert ``time`` into :attr:`TimeUnit.BATCH`. Requires that ``time.unit == TimeUnit.EPOCH``.

    Args:
        batch_size (int): The optimization batch size.
        drop_last (bool): Whether the dataloader is dropping last (incomplete) batches.
        dataset_num_samples (int): The number of samples in the dataset.

    Raises:
        RuntimeError: Raised if ``time.unit != TimeUnit.EPOCH``

    Returns:
        Time: The time, in :attr:`TimeUnit.BATCH`.
    """
    if time.unit != TimeUnit.EPOCH:
        raise RuntimeError(f"Time {time} units are not epochs.")
    if drop_last:
        num_batches_per_epoch = dataset_num_samples // batch_size
    else:
        num_batches_per_epoch = (dataset_num_samples - 1) // batch_size + 1
    return Time(time.value * num_batches_per_epoch, TimeUnit.BATCH)


def convert_epoch_to_sample(time,
                            *,
                            drop_last: bool,
                            dataset_num_samples: int,
                            batch_size: Optional[int] = None) -> Time:
    """Convert ``time`` into :attr:`TimeUnit.SAMPLE`. Requires that ``time.unit == TimeUnit.EPOCH``.

    Args:
        drop_last (bool): Whether the dataloader is dropping last (incomplete) batches.
        dataset_num_samples (int): The number of samples in the dataset.
        batch_size (int, optional): The optimization batch size. Required if ``drop_last`` is ``True``.

    Raises:
        RuntimeError: Raised if ``time.unit != TimeUnit.EPOCH``

    Returns:
        Time: The time, in :attr:`TimeUnit.SAMPLE`.
    """
    if time.unit != TimeUnit.EPOCH:
        raise RuntimeError(f"Time {time} units are not epochs.")
    if drop_last:
        if batch_size is None:
            raise ValueError("batch_size is required to convert from EPOCH to SAMPLE when drop_last=True")
        num_batches_per_epoch = dataset_num_samples // batch_size
        num_samples_per_epoch = num_batches_per_epoch * batch_size
        return Time(time.value * num_samples_per_epoch, TimeUnit.SAMPLE)
    else:
        return Time(time.value * dataset_num_samples, TimeUnit.SAMPLE)


def convert_epoch_to_token(time, *, dataset_num_tokens: int) -> Time:
    """Convert ``time`` into :attr:`TimeUnit.TOKEN`. Requires that ``time.unit == TimeUnit.EPOCH``.

    .. note::

        The conversion is valid only if the dataloader yields all batches (i.e. ``drop_last`` is ``False``).

    Args:
        dataset_num_tokens (int): The number of tokens in the dataset.

    Raises:
        RuntimeError: Raised if ``time.unit != TimeUnit.EPOCH``

    Returns:
        Time: The time, in :attr:`TimeUnit.TOKEN`.
    """
    if time.unit != TimeUnit.EPOCH:
        raise RuntimeError(f"Time {time} units are not epochs.")
    return Time(time.value * dataset_num_tokens, TimeUnit.TOKEN)


def convert_batch_to_epoch(time, *, batch_size: int, drop_last: bool, dataset_num_samples: int):
    """Convert ``time`` into :attr:`TimeUnit.EPOCH`. Requires that ``time.unit == TimeUnit.BATCH``.

    Args:
        batch_size (int): The optimization batch size.
        drop_last (bool): Whether the dataloader is dropping last (incomplete) batches.
        dataset_num_samples (int): The number of samples in the dataset.

    Raises:
        RuntimeError: Raised if ``time.unit != TimeUnit.BATCH``

    Returns:
        Time: The time, in :attr:`TimeUnit.EPOCH`.
    """
    if time.unit != TimeUnit.BATCH:
        raise RuntimeError(f"Time {time} units are not batches.")
    if drop_last:
        num_batches_per_epoch = dataset_num_samples // batch_size
    else:
        num_batches_per_epoch = (dataset_num_samples - 1) // batch_size + 1
    return Time(time.value // num_batches_per_epoch, TimeUnit.EPOCH)


def convert_batch_to_sample(
    time,
    *,
    batch_size: int,
    drop_last: bool,
    dataset_num_samples: Optional[int] = None,
) -> Time:
    """Convert ``time`` into :attr:`TimeUnit.SAMPLE`. Requires that ``time.unit == TimeUnit.BATCH``.

    Args:
        batch_size (int): The optimization batch size.
        drop_last (bool): Whether the dataloader is dropping last (incomplete) batches.
        dataset_num_samples (int, optional): The number of samples in the dataset. Required if ``drop_last`` is ``False``.

    Raises:
        RuntimeError: Raised if ``time.unit != TimeUnit.BATCH``

    Returns:
        Time: The time, in :attr:`TimeUnit.SAMPLE`.
    """
    if time.unit != TimeUnit.BATCH:
        raise RuntimeError(f"Time {time} units are not batches.")
    if drop_last:
        return Time(time.value * batch_size, TimeUnit.SAMPLE)
    else:
        if dataset_num_samples is None:
            raise ValueError("dataset_num_samples is required to convert from EPOCH to BATCH when drop_last=False")
        num_batches_per_epoch = (dataset_num_samples - 1) // batch_size + 1
        samples_from_complete_epochs = (time.value // num_batches_per_epoch) * dataset_num_samples
        remaining_batches = time.value % num_batches_per_epoch
        samples_from_incomplete_epochs = remaining_batches * batch_size
        return Time(samples_from_complete_epochs + samples_from_incomplete_epochs, TimeUnit.SAMPLE)


def convert_sample_to_epoch(time,
                            *,
                            drop_last: bool,
                            dataset_num_samples: int,
                            batch_size: Optional[int] = None) -> Time:
    """Convert ``time`` into :attr:`TimeUnit.EPOCH`. Requires that ``time.unit == TimeUnit.SAMPLE``.

    Args:
        drop_last (bool): Whether the dataloader is dropping last (incomplete) batches.
        dataset_num_samples (int): The number of samples in the dataset.
        batch_size (int, optional): The optimization batch size. Required if ``drop_last`` is ``True``.

    Raises:
        RuntimeError: Raised if ``time.unit != TimeUnit.SAMPLE``

    Returns:
        Time: The time, in :attr:`TimeUnit.EPOCH`.
    """
    if time.unit != TimeUnit.SAMPLE:
        raise RuntimeError(f"Time {time} units are not samples.")
    if drop_last:
        if batch_size is None:
            raise ValueError("batch_size is required to convert from SAMPLE to SAMPLE when drop_last=True")
        num_batches_per_epoch = dataset_num_samples // batch_size
        num_samples_per_epoch = num_batches_per_epoch * batch_size
        return Time(time.value // num_samples_per_epoch, TimeUnit.EPOCH)
    else:
        return Time(time.value // dataset_num_samples, TimeUnit.EPOCH)


def convert_sample_to_batch(
    time,
    *,
    batch_size: int,
    drop_last: bool,
    dataset_num_samples: Optional[int] = None,
) -> Time:
    """Convert ``time`` into :attr:`TimeUnit.BATCH`. Requires that ``time.unit == TimeUnit.SAMPLE``.

    Args:
        batch_size (int): The optimization batch size.
        drop_last (bool): Whether the dataloader is dropping last (incomplete) batches.
        dataset_num_samples (int, optional): The number of samples in the dataset. Required if ``drop_last`` is ``False``.

    Raises:
        RuntimeError: Raised if ``time.unit != TimeUnit.SAMPLE``

    Returns:
        Time: The time, in :attr:`TimeUnit.BATCH`.
    """
    if time.unit != TimeUnit.SAMPLE:
        raise RuntimeError(f"Time {time} units are not samples.")
    if drop_last:
        return Time(time.value // batch_size, TimeUnit.BATCH)
    else:
        if dataset_num_samples is None:
            raise ValueError("dataset_num_samples is required to convert from SAMPLE to BATCH when drop_last=False")
        num_batches_per_epoch = (dataset_num_samples - 1) // batch_size + 1
        batches_from_complete_epochs = (time.value // dataset_num_samples) * num_batches_per_epoch
        batches_from_incomplete_epoch = (time.value % dataset_num_samples) // batch_size
        return Time(batches_from_complete_epochs + batches_from_incomplete_epoch, TimeUnit.BATCH)


def convert_token_to_epoch(time, *, dataset_num_tokens: int) -> Time:
    """Convert ``time`` into :attr:`TimeUnit.EPOCH`. Requires that ``time.unit == TimeUnit.TOKEN``.

    .. note::

        The conversion is valid only if the dataloader yields all batches (i.e. ``drop_last`` == ``False``).

    Args:
        dataset_num_tokens (int): The number of tokens in the dataset.

    Raises:
        RuntimeError: Raised if ``time.unit != TimeUnit.TOKEN``

    Returns:
        Time: The time, in :attr:`TimeUnit.EPOCH`.
    """
    if time.unit != TimeUnit.TOKEN:
        raise RuntimeError(f"Time {time} units are not tokens.")
    return Time(time.value // dataset_num_tokens, TimeUnit.EPOCH)


def convert_to_duration(time, *, max_training_duration: Union[str, Time]) -> Time:
    """Convert ``time`` into :attr:`TimeUnit.DURATION`.

    Args:
        max_training_duration (str or Time): The total training duration.

    Returns:
        Time: The time, in :attr:`TimeUnit.DURATION`.
    """
    if time.unit == TimeUnit.DURATION:
        return Time(time.value, time.unit)
    if isinstance(max_training_duration, str):
        max_training_duration = Time.from_timestring(max_training_duration)
    return time / max_training_duration


def convert_from_duration(time, *, max_training_duration: Union[str, Time]) -> Time:
    """Convert ``time`` into the units of ``max_training_duration``.

    Args:
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
