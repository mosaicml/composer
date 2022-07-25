# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""A collection of classes to store GLUE metrics during finetuning using the NLP entrypoint."""

from dataclasses import dataclass
from typing import Any, Dict, List

__all__ = ['GlueState', 'GLUEMetricsLogger']


class GLUEMetricsLogger:
    """Class mapping all GLUE tasks to their respective average metric values.

    Args:
        tasks_to_avg_metric dict(str, Any): dictionary mapping GLUE task names to their avg values.

    """
    task_to_avg_metric: Dict[str, Any]

    def __init__(self, task_names) -> None:
        self.task_to_avg_metric = {}

        for task in task_names:
            self.task_to_avg_metric[task] = None


@dataclass
class GlueState:
    """Class storing all GLUE metrics per checkpoint collected during a finetuning job spawned by the NLP entrypoint.

    This class maps checkpoint names to GLUEMetricsLogger instances which map tasks to their respective average
    metric values.

    Args:
        task_names list(str): the names of the GLUE tasks stored in the data struct
        ckpt_to_tasks dict(str, GLUEMetricsLogger): dictionary mapping checkpoint names to GLUEMetricsLogger instances
    """
    task_names: List[str]
    ckpt_to_tasks: Dict[str, GLUEMetricsLogger]
