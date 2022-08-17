# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Hyperparameters for the :class:`torchmetrics.Metric`."""

from __future__ import annotations

import abc
import logging
import textwrap
from dataclasses import asdict, dataclass
from typing import Optional, Type

import yahp as hp
from torchmetrics import Metric

log = logging.getLogger(__name__)

__all__ = ['MetricHparams']


@dataclass
class MetricHparams(hp.Hparams, abc.ABC):
    """Base class for :class:`torchmetrics.Metric` hparams classes.

    Args:
        name (str): Name for metric.
        dist_sync_on_step (bool, optional): If metric state should synchronize on
            `forward()`. Default: ``False``.
        compute_on_cpu (bool, optional): If metric state should be stored on CPU during
            computations. Only works for list states. Default: ``False``.
    """

    name: str = hp.required(doc='Name for metric')
    dist_sync_on_step: bool = hp.optional(doc='If metric state should synchronize on forward()', default=False)
    compute_on_cpu: Optional[bool] = hp.optional(doc=textwrap.dedent("""If metric state should be stored on CPU during
                            computations. Only works for list states."""),
                                                 default=False)
    metric_class = None  # type: Optional[Type[Metric]]

    def initialize_object(self) -> Metric:
        """Initializes a :class:`torchmetrics.Metric`.

        Args:

        Returns:
            Metric: a :class:`torchmetrics.Metric`
        """
        if self.metric_class is None:
            raise ValueError(f'{type(self).__name__}.metric_class must be defined')
        return self.metric_class(**asdict(self))
