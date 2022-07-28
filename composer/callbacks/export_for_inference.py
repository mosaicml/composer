# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Callback to export model for inference."""

from __future__ import annotations

import logging
from typing import Callable, Optional, Sequence, Union

import torch.nn as nn

from composer.core import State
from composer.core.callback import Callback
from composer.loggers import Logger
from composer.utils.inference import ExportFormat, export_for_inference
from composer.utils.object_store import ObjectStore

log = logging.getLogger(__name__)

__all__ = ['ExportForInference']


class ExportForInference(Callback):  # noqa: D101
    """Callback to export model for inference.

    Args:
        save_format (Union[str, ExportFormat]):  Format to export to. Either ``"torchscript"`` or ``"onnx"``.
        out_filename (str): The path for storing the exported model. It can be a path to a file on the local disk,
            a URL, or if ``save_object_store`` is set, the object name
            in a cloud bucket. For example, ``my_run/exported_model``.
        save_object_store (ObjectStore, optional): If the ``out_filename`` is in an object name in a cloud bucket
            (i.e. AWS S3 or Google Cloud Storage), an instance of
            :class:`~.ObjectStore` which will be used
            to store the exported model. Set this to ``None`` if ``out_filename`` is a local filepath.
            (default: ``None``)
        transforms (Union[Callable, Sequence[Callable]], optional): transformations (usually optimizations) that should
            be applied to the model. Each should be a callable that takes a model and returns a modified model.
    """

    def __init__(
        self,
        save_format: Union[str, ExportFormat],
        out_filename: str,
        save_object_store: Optional[ObjectStore] = None,
        transforms: Optional[Union[Callable[[nn.Module], nn.Module], Sequence[Callable[[nn.Module],
                                                                                       nn.Module]]]] = None,
    ):
        self.save_format = save_format
        self.out_filename = out_filename
        self.save_object_store = save_object_store
        self.transforms = transforms

    def fit_end(self, state: State, logger: Logger):
        del logger
        self.export_model(state)

    def export_model(self, state: State):
        if state.dataloader is not None:
            sample_input = next(iter(state.dataloader))
            export_for_inference(model=state.model,
                                 save_format=self.save_format,
                                 save_path=self.out_filename,
                                 save_object_store=self.save_object_store,
                                 sample_input=(sample_input,),
                                 transforms=self.transforms)
        else:
            raise RuntimeError('Exporing model failed because state has no dataloader.')
