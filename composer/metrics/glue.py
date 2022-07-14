# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""A collecion of classes to store GLUE metrics during finetuning using the NLP entrypoint. """

from dataclasses import dataclass
from typing import Any, Dict, List

__all__ = ['GlueState', 'GLUEMetricsLogger']

@dataclass
class GLUEMetricsLogger:
    """ Class mapping all GLUE tasks to their respective average metric values. 

    Args:
        tasks_to_avg_metric dict(str, Any): dictionary mapping GLUE task names to their avg values.
    
    """
    task_to_avg_metric: Dict[str, Any]  

    def __init__(self, task_names) -> None:
        self.tasks_to_avg_metric = {}
        for task in task_names:
            self.task_to_avg_metric[task] = None
    
@dataclass
class GlueState:
    """ Class storing all GLUE metrics per checkpoint collected during a finetuning job spawned by the NLP entrypoint. 
    This class maps checkpoint names to GLUEMetricsLogger instances which map tasks to their respective average 
    metric values. 
    
    Args: 
        task_names list(str): the names of the GLUE tasks stored in the data struct
        ckpt_to_tasks dict(str, GLUEMetricsLogger): dictionary mapping checkpoint names to GLUEMetricsLogger instances
    """
    task_names: List[str]
    ckpt_to_tasks: Dict[str, GLUEMetricsLogger]
        
    def __init__(self, task_names, ckpt_to_tasks) -> None:
        self.task_names = task_names
        self.ckpt_to_tasks = ckpt_to_tasks

    def init_ckpt_dict(self, ckpt_name: str):
        """ Create an empty GLUEMetricsLogger instance with all the tasks for a new checkpoint. """
        self.ckpt_to_tasks[ckpt_name] = GLUEMetricsLogger(self.task_names)

    def get_logged_ckpts(self) -> List[str]:
        """ Get the list of currently logged checkpoints. """
        return list(self.ckpt_to_tasks.keys())
    
    def set_task_metric(self, ckpt_name, task_name, metric) -> None:
        """" Add a new metric value to the average value for a given task and checkpoint. """
        self.ckpt_to_tasks[ckpt_name].task_to_avg_metric[task_name] = metric

    def get_task_metric(self, ckpt_name, task_name) -> int:
        """ Get the average metric value for a given task and checkpoint. """
        return self.ckpt_to_tasks[ckpt_name].task_to_avg_metric[task_name]
