# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Callback to export model for inference."""

from __future__ import annotations

import logging
from copy import deepcopy
from typing import Any, Optional, Sequence, Union

import torch.nn as nn

from composer.core import Callback, State
from composer.loggers import Logger
from composer.utils import ExportFormat, ObjectStore, Transform, export_with_logger

log = logging.getLogger(__name__)

__all__ = ['ExportForInferenceCallback']


class ExportForInferenceCallback(Callback):
    """Callback to export model for inference.

    Example:
        .. doctest::

            >>> from composer import Trainer
            >>> from composer.callbacks import ExportForInferenceCallback
            >>> # constructing trainer object with this callback
            >>> trainer = Trainer(
            ...     model=model,
            ...     train_dataloader=train_dataloader,
            ...     eval_dataloader=eval_dataloader,
            ...     optimizers=optimizer,
            ...     max_duration="1ep",
            ...     callbacks=[ExportForInferenceCallback(save_format='torchscript',save_path='/tmp/model.pth')],
            ... )

    Args:
        save_format (Union[str, ExportFormat]):  Format to export to. Either ``"torchscript"`` or ``"onnx"``.
        save_path (str): The path for storing the exported model. It can be a path to a file on the local disk,
            a URL, or if ``save_object_store`` is set, the object name
            in a cloud bucket. For example, ``my_run/exported_model``.
        save_object_store (ObjectStore, optional): If the ``save_path`` is in an object name in a cloud bucket
            (i.e. AWS S3 or Google Cloud Storage), an instance of
            :class:`~.ObjectStore` which will be used
            to store the exported model. If this is set to ``None``,  will save to ``save_path`` using the logger.
            (default: ``None``)
        sample_input (Any, optional): Example model inputs used for tracing. This is needed for "onnx" export
        transforms (Sequence[Transform], optional): transformations (usually optimizations) that should
            be applied to the model. Each Transform should be a callable that takes a model and returns a modified model.
    """

    def __init__(
        self,
        save_format: Union[str, ExportFormat],
        save_path: str,
        save_object_store: Optional[ObjectStore] = None,
        sample_input: Optional[Any] = None,
        transforms: Optional[Sequence[Transform]] = None,
    ):
        self.save_format = save_format
        self.save_path = save_path
        self.save_object_store = save_object_store
        self.sample_input = sample_input
        self.transforms = transforms

    def after_dataloader(self, state: State, logger: Logger) -> None:
        del logger
        if self.sample_input is None and self.save_format == 'onnx':
            self.sample_input = deepcopy(state.batch)

    def fit_end(self, state: State, logger: Logger):
        self.export_model(state, logger)

    def export_model(self, state: State, logger: Logger):
        export_model = state.model.module if state.is_model_ddp else state.model
        if not isinstance(export_model, nn.Module):
            raise ValueError(f'Exporting Model requires type torch.nn.Module, got {type(export_model)}')
        export_with_logger(model=export_model,
                           save_format=self.save_format,
                           save_path=self.save_path,
                           logger=logger,
                           save_object_store=self.save_object_store,
                           sample_input=(self.sample_input, {}),
                           transforms=self.transforms)
