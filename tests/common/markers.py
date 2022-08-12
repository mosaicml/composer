# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Pytest marker helpers."""

from typing import Callable

import pytest

from composer.core import Precision


def device(*args, precision=False):
    """Decorator for device and optionally precision.

    Input choices are ('cpu', 'gpu'), or if precision=True,
    also accept ('gpu-amp', 'gpu-fp32', and 'cpu-fp32').

    Returns the parameter "device", or if precision=True,
    also returns the parameter "precision".
    """
    # convert cpu-fp32 and gpu-fp32 to cpu, gpu
    if not precision and any(['-' in arg for arg in args]):
        raise ValueError('-fp32 and -amp tags must be removed if precision=False')
    args = [arg.replace('-fp32', '') for arg in args]

    if precision:
        devices = {
            'cpu': pytest.param('cpu', Precision.FP32, id='cpu-fp32'),
            'gpu': pytest.param('gpu', Precision.FP32, id='gpu-fp32', marks=pytest.mark.gpu),
            'gpu-amp': pytest.param('gpu', Precision.AMP, id='gpu-amp', marks=pytest.mark.gpu)
        }
        name = 'device,precision'
    else:
        devices = {
            'cpu': pytest.param('cpu', id='cpu'),
            'gpu': pytest.param('gpu', id='gpu', marks=pytest.mark.gpu),
        }
        name = 'device'

    parameters = [devices[arg] for arg in args]

    def decorator(test):
        if not parameters:
            return test
        return pytest.mark.parametrize(name, parameters)(test)

    return decorator


def world_size(*world_sizes: int, param_name: str = 'world_size'):
    """Decorator to mark tests with a given world size. This helper automatically sets the `pytest.mark.world_size`
    marker.

    Args:
        world_sizes (int): The world sizes.
        param_name (str, optional): The parameter name for the `world_size` parameter. Defaults to ``'world_size'``.

    Example:
    >>> @world_size(1, 2)
    def test_something(world_size: int):
        ...
    """

    parameters = []
    for world_size in world_sizes:
        if world_size == 1:
            parameters.append(pytest.param(1))
        else:
            parameters.append(pytest.param(2, marks=pytest.mark.world_size(2)))

    def decorator(test: Callable):
        if len(parameters) == 0:
            return test

        return pytest.mark.parametrize(param_name, parameters)(test)

    return decorator
