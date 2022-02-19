# Copyright 2021 MosaicML. All Rights Reserved.

"""Core Layer Freezing classes and functions."""

from __future__ import annotations

import logging
import textwrap
from typing import List, Optional, Tuple

import torch

from composer.core import Algorithm, Event, Logger, State
from composer.core.types import Model, Optimizers

log = logging.getLogger(__name__)

__all__ = ["LayerFreezing", "freeze_layers"]


def freeze_layers(
    model: Model,
    optimizers: Optimizers,
    current_duration: float,
    freeze_start: float,
    freeze_level: float,
) -> Tuple[int, float]:
    """Progressively freeze the layers of the network in-place during training, starting with the earlier layers.

    Example:
         .. testcode::

            from composer.algorithms.layer_freezing import freeze_layers
            freeze_depth, feeze_level = freeze_layers(
                                            model=model,
                                            optimizers=optimizer,
                                            current_duration=0.5,
                                            freeze_start=0.0,
                                            freeze_level=1.0
                                        )


    Args:
        model (Model): The model being trained.
        optimizers (Optimizers): The optimizers used during training.
        current_duration (float): The fraction on [0; 1) of the training process complete.
        freeze_start (float): The fraction of the training process on [0; 1) to run before freezing begins.
        freeze_level (float): The maximum fraction of layers on [0; 1) to freeze.

    Return:
        (int, float): The number of layers frozen, and the percentage of the total model frozen.
    """
    # Flatten out the layers
    flat_children = []
    _get_layers(model, flat_children)
    # Determine how many layers to freeze
    freeze_percentage = _freeze_schedule(current_duration=current_duration,
                                         freeze_start=freeze_start,
                                         freeze_level=freeze_level)
    freeze_depth = int(freeze_percentage * len(flat_children[0:-1]))

    # Freeze the parameters in the chosen layers
    for i, child in enumerate(flat_children[0:-1]):
        if i < freeze_depth:
            for p in child.parameters():
                _remove_param_from_optimizers(p, optimizers)
                # Do not compute gradients for this param.
                p.requires_grad = False

    # Log results
    log.info(
        textwrap.dedent(f"""\
            Applied Layer Freezing with freeze_start={freeze_start},
            freeze_level={freeze_level}. Froze {freeze_depth} layers in the model which
            equates to {freeze_percentage * 100}% of all layers."""))
    return freeze_depth, freeze_percentage


class LayerFreezing(Algorithm):
    """Progressively freeze the layers of the network during training, starting with the earlier layers.

    Freezing starts after the fraction of training specified by ``freeze_start``
    has elapsed. The fraction of layers frozen increases linearly until it
    reaches ``freeze_level`` at the end of training.

    This freezing schedule is most similar to
    `FreezeOut <https://arxiv.org/abs/1706.04983>`_ and
    `Freeze Training <https://arxiv.org/abs/1706.05806>`_.

    Runs on ``Event.EPOCH_END``.

    Example:
         .. testcode::

            from composer.algorithms import LayerFreezing
            from composer.trainer import Trainer
            layer_freezing_algorithm = LayerFreezing(freeze_start=0.0, freeze_level=1.0)
            trainer = Trainer(
                model=model,
                train_dataloader=train_dataloader,
                eval_dataloader=eval_dataloader,
                max_duration="1ep",
                algorithms=[layer_freezing_algorithm],
                optimizers=[optimizer]
            )

    Args:
        freeze_start (float): The fraction of training to run before freezing begins.
        freeze_level (float): The maximum fraction of layers to freeze.
    """

    def __init__(self, freeze_start: float = 0.5, freeze_level: float = 1.0):
        self.freeze_start = freeze_start
        self.freeze_level = freeze_level

    @property
    def find_unused_parameters(self) -> bool:
        """Override in order to tell DDP that some parameters will not have gradients computed for them after layer
        freezing is applied."""
        return True

    def match(self, event: Event, state: State) -> bool:
        """Run on ``Event.EPOCH_END``."""
        del state  # unused
        return event == Event.EPOCH_END

    def apply(self, event: Event, state: State, logger: Logger) -> Optional[int]:
        """Freeze layers in the model."""
        del event  # unused
        optimizers = state.optimizers
        assert optimizers is not None
        freeze_depth, freeze_percentage = freeze_layers(
            model=state.model,
            optimizers=optimizers,
            current_duration=float(state.get_elapsed_duration()),
            freeze_start=self.freeze_start,
            freeze_level=self.freeze_level,
        )
        logger.metric_epoch({
            'layer_freezing/layers_frozen': freeze_depth,
            'layer_freezing/percentage_frozen': freeze_percentage
        })


def _freeze_schedule(current_duration: float, freeze_start: float, freeze_level: float) -> float:
    """Implements a linear schedule for freezing. The schedule is linear and begins with no freezing and linearly
    increases the fraction of layers frozen, reaching the fraction specified by 'freeze_level' at the end of training.
    The start of freezing is given as a fraction of the total training duration, and is set with 'freeze_start'.

    Args:
        current_duration (float): The elapsed training duration.
        freeze_start (float): The fraction of training to run before freezing begins.
        freeze_level (float): The maximum fraction of levels to freeze.
    """
    # No freezing if the current epoch is less than this
    if current_duration <= freeze_start:
        return 0.0
    # `Calculate the total time for freezing to occur
    total_freezing_time = 1.0 - freeze_start
    # Calculate the amount of freezing time that has elapsed
    freezing_time_elapsed = current_duration - freeze_start
    # Calculate the fraction of the freezing time elapsed.
    freezing_time_elapsed_frac = freezing_time_elapsed / total_freezing_time
    # Scale this fraction by the amount of freezing to do.
    return freeze_level * freezing_time_elapsed_frac


def _get_layers(module: Model, flat_children: List[Model]):
    """Helper function to get all submodules.

    Does a depth first search to flatten out modules which
    contain parameters.

    Args:
        module (Model): Current module to search.
        flat_children (List[Model]): List containing modules.
    """
    # Check if given module has no children and parameters.
    if (len(list(module.children())) == 0 and len(list(module.parameters())) > 0):
        flat_children.append(module)
    else:
        # Otherwise, continue the search over its children.
        for child in module.children():
            _get_layers(child, flat_children)


def _remove_param_from_optimizers(p: torch.nn.Parameter, optimizers: Optimizers):
    """Helper function to freeze the training of a parameter.

    To freeze a parameter, it must be removed from the optimizer,
    otherwise momentum and weight decay may still be applied.

    Args:
        p (torch.nn.Parameter): The parameter being frozen.
        optimizers (Optimizers): The optimizers used during training.
    """
    # Force optimizers to be iterable
    if not isinstance(optimizers, (list, tuple)):
        optimizers = [optimizers]

    # Search over params in the optimizers to find and remove the
    # given param. Necessary due to the way params are stored.
    for optimizer in optimizers:
        for group in optimizer.param_groups:
            group['params'] = list(filter(lambda x: id(x) != id(p), group['params']))
