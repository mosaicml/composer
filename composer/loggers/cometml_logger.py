# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Log to `Comet <https://www.comet.com/?utm_source=mosaicml&utm_medium=partner&utm_campaign=mosaicml_comet_integration>`_."""

from __future__ import annotations

import textwrap
from typing import Any, Dict, Optional, Sequence, Union

import numpy as np
import torch
from torch import nn
from torchvision.utils import draw_segmentation_masks

from composer.core.state import State
from composer.loggers.logger import Logger
from composer.loggers.logger_destination import LoggerDestination
from composer.utils import MissingConditionalImportError, dist

__all__ = ['CometMLLogger']


class CometMLLogger(LoggerDestination):
    """Log to `Comet <https://www.comet.com/?utm_source=mosaicml&utm_medium=partner&utm_campaign=mosaicml_comet_integration>`_.

    Args:
        workspace (str, optional): The name of the workspace which contains the project
            you want to attach your experiment to. If nothing specified will default to your
            default workspace as configured in your comet account settings.
        project_name (str, optional): The name of the project to categorize your experiment in.
            A new project with this name will be created under the Comet workspace if one
            with this name does not exist. If no project name specified, the experiment will go
            under Uncategorized Experiments.
        log_code (bool): Whether to log your code in your experiment (default: ``False``).
        log_graph (bool): Whether to log your computational graph in your experiment
            (default: ``False``).
        name (str, optional): The name of your experiment. If not specified, it will be set
            to :attr:`.State.run_name`.
        rank_zero_only (bool, optional): Whether to log only on the rank-zero process.
            (default: ``True``).
        exp_kwargs (Dict[str, Any], optional): Any additional kwargs to
            comet_ml.Experiment(see
            `Comet documentation <https://www.comet.com/docs/v2/api-and-sdk/python-sdk/reference/Experiment/?utm_source=mosaicml&utm_medium=partner&utm_campaign=mosaicml_comet_integration>`_).
    """

    def __init__(
        self,
        workspace: Optional[str] = None,
        project_name: Optional[str] = None,
        log_code: bool = False,
        log_graph: bool = False,
        name: Optional[str] = None,
        rank_zero_only: bool = True,
        exp_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        try:
            from comet_ml import Experiment
        except ImportError as e:
            raise MissingConditionalImportError(extra_deps_group='comet_ml',
                                                conda_package='comet_ml',
                                                conda_channel='conda-forge') from e

        self._enabled = (not rank_zero_only) or dist.get_global_rank() == 0

        if exp_kwargs is None:
            exp_kwargs = {}

        if workspace is not None:
            exp_kwargs['workspace'] = workspace

        if project_name is not None:
            exp_kwargs['project_name'] = project_name

        exp_kwargs['log_code'] = log_code
        exp_kwargs['log_graph'] = log_graph

        self.name = name
        self._rank_zero_only = rank_zero_only
        self._exp_kwargs = exp_kwargs
        self.experiment = None
        if self._enabled:
            self.experiment = Experiment(**self._exp_kwargs)
            self.experiment.log_other('Created from', 'mosaicml-composer')

    def init(self, state: State, logger: Logger) -> None:
        del logger  # unused

        # Use the logger run name if the name is not set.
        if self.name is None:
            self.name = state.run_name

        # Adjust name and group based on `rank_zero_only`.
        if not self._rank_zero_only:
            self.name += f'-rank{dist.get_global_rank()}'

        if self._enabled:
            assert self.experiment is not None
            self.experiment.set_name(self.name)

    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        if self._enabled:
            assert self.experiment is not None
            self.experiment.log_metrics(dic=metrics, step=step)

    def log_hyperparameters(self, hyperparameters: Dict[str, Any]):
        if self._enabled:
            assert self.experiment is not None
            self.experiment.log_parameters(hyperparameters)

    def log_images(self,
                   images: Union[np.ndarray, torch.Tensor, Sequence[Union[np.ndarray, torch.Tensor]]],
                   name: str = 'Image',
                   channels_last: bool = False,
                   step: Optional[int] = None,
                   masks: Optional[Dict[str, Union[np.ndarray, torch.Tensor, Sequence[Union[np.ndarray,
                                                                                            torch.Tensor]]]]] = None,
                   mask_class_labels: Optional[Dict[int, str]] = None,
                   use_table: bool = True):

        del use_table, mask_class_labels  # Unused (only for wandb)
        if self._enabled:
            image_channels = 'last' if channels_last else 'first'
            # Convert to singleton sequences if a single image or mask is specified.
            if not isinstance(images, Sequence) and images.ndim <= 3:
                images = [images]

            # For pyright.
            assert self.experiment is not None

            if masks is not None:
                for mask_name, mask_tensor in masks.items():
                    if not isinstance(mask_tensor, Sequence) and mask_tensor.ndim == 2:
                        masks[mask_name] = [mask_tensor]
                mask_names = list(masks.keys())
                for index, (image, *mask_set) in enumerate(zip(images, *masks.values())):
                    # Log input image
                    comet_image = _convert_to_comet_image(image)
                    self.experiment.log_image(comet_image,
                                              name=f'{name}_{index}',
                                              image_channels=image_channels,
                                              step=step)

                    # Convert 2D index mask to one-hot boolean mask.
                    mask_set = [_convert_to_comet_mask(mask) for mask in mask_set]

                    # Log input image with mask overlay and mask by itself for each type of mask.
                    for mask_name, mask in zip(mask_names, mask_set):
                        if channels_last:
                            # permute to channels_first to be compatible with draw_segmentation_masks.
                            comet_image = image.permute(2, 0, 1)
                        # Log input image with mask superimposed.
                        im_with_mask_overlay = draw_segmentation_masks(comet_image.to(torch.uint8), mask, alpha=0.6)
                        self.experiment.log_image(im_with_mask_overlay,
                                                  name=f'{name}_{index} + {mask_name} mask overlaid',
                                                  image_channels='first',
                                                  step=step)
                        # Log mask only.
                        mask_only = draw_segmentation_masks(torch.zeros_like(comet_image.to(torch.uint8)), mask)
                        self.experiment.log_image(mask_only,
                                                  name=f'{mask_name}_{index} mask',
                                                  step=step,
                                                  image_channels='first')
            else:
                for index, image in enumerate(images):
                    comet_image = _convert_to_comet_image(image)
                    self.experiment.log_image(comet_image,
                                              name=f'{name}_{index}',
                                              image_channels=image_channels,
                                              step=step)

    def post_close(self):
        if self._enabled:
            assert self.experiment is not None
            self.experiment.end()


def _convert_to_comet_image(image: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
    if isinstance(image, torch.Tensor):
        image = image.data.cpu()
    elif isinstance(image, np.ndarray):
        image = torch.from_numpy(image)
    # Error out for empty arrays or weird arrays of dimension 0.
    if np.any(np.equal(image.shape, 0)):
        raise ValueError(f'Got an image (shape {image.shape}) with at least one dimension being 0! ')
    image = image.squeeze()
    if image.ndim > 3:
        raise ValueError(
            textwrap.dedent(f'''Input image must be 1, 2, or 3 dimensions, but instead got
                            {image.ndim} dims at shape: {image.shape} Your input image was
                             interpreted as a batch of {image.ndim}-dimensional images
                             because you either specified a {image.ndim + 1}D image or a
                             list of {image.ndim}D images. Please specify either a 4D
                             image of a list of 3D images'''))

    return image


def _convert_to_comet_mask(mask: Union[np.ndarray, torch.Tensor]):
    if isinstance(mask, np.ndarray):
        mask = torch.from_numpy(mask)
    mask = mask.squeeze()
    if mask.ndim != 2:
        raise ValueError(
            textwrap.dedent(f'''Each input mask must be 2 dimensions, but instead got
                                {mask.ndim} dims at shape: {mask.shape}. Please specify
                                a sequence of 2D masks or 3D batch of 2D masks .'''))

    num_classes = int(torch.max(mask)) + 1
    one_hot_mask = nn.functional.one_hot(mask, num_classes).permute(2, 0, 1).bool()
    return one_hot_mask
