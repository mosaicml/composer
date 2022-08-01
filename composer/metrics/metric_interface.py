# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Protocol

__all__ = ['MetricInterface']


class MetricInterface(Protocol):
    """Protocol to describe a torchmetrics.Metric.
	All torchmetrics would be compatible with this interface, but it allows
	for custom non-torchmetrics, so long as the metric class implements
	this protocol. The user would never have to import this class;
	it is used for type checking only.
	"""

    def reset(self) -> None:
        """Reset the metric"""

    def compute(self) -> Any:
        """Compute the metric"""

    def update(self, *args: Any, **kwargs: Any) -> Any:
        """Update the metrics"""
