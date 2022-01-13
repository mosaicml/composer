# Copyright 2021 MosaicML. All Rights Reserved.

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Optional

import yahp as hp

from composer.algorithms import AlgorithmHparams
from composer.core import Algorithm, Event, Logger, State
from composer.core.types import Model, Optimizers

log = logging.getLogger(__name__)


@dataclass
class LayerFreezingHparams(AlgorithmHparams):
    """See :class:`LayerFreezing`"""

    freeze_start: float = hp.optional(doc='The percentage of epochs to run before freezing begins.', default=0.5)
    freeze_level: float = hp.optional(doc='Scale factor for the percentage of the network to freeze.', default=1.0)

    def initialize_object(self) -> LayerFreezing:
        return LayerFreezing(**asdict(self))


def _freeze_schedule(current_epoch: int, max_epochs: int, freeze_start: float, freeze_level: float):
    """Implements a linear schedule for freezing.

    The schedule is linear and begins with no freezing and
    linearly increases the fraction of layers frozen, reaching
    the fraction specified by 'freeze_level' at the final epoch.
    The start of freezing is given as a fraction of the total
    number of epochs, and is set with 'freeze_start'.

    Args:
        current_epoch: Integer specifying the current epoch.
        max_epochs: The max number of epochs training will run for.
        freeze_start: The fraction of epochs to run before freezing begins.
        freeze_level: The maximum fraction of levels to freeze.
    """
    # Calculate the epoch to start freezing
    freeze_start_epoch = int(freeze_start * max_epochs)
    # No freezing if the current epoch is less than this
    if current_epoch <= freeze_start_epoch:
        freeze_percentage = 0
    else:
        # Calculate the total time for freezing to occur
        reduced_time = max_epochs - freeze_start_epoch
        # Calculate the amount of freezing time that has elapsed
        time_elapsed = current_epoch - freeze_start_epoch
        # Calculate the fraction of the freezing time elapsed.
        scaling = time_elapsed / reduced_time
        # Scale this fraction by the amount of freezing to do.
        freeze_percentage = freeze_level * scaling

    return freeze_percentage


def _get_layers(module, flat_children):
    """Helper function to get all submodules.

    Does a depth first search to flatten out modules which
    contain parameters.

    Args:
        module: Current module to search.
        flat_children: List containing modules.
    """
    # Check if given module has no children and parameters.
    if (len(list(module.children())) == 0 and len(list(module.parameters())) > 0):
        flat_children.append(module)
    else:
        # Otherwise, continue the search over its children.
        for child in module.children():
            _get_layers(child, flat_children)


def _remove_param_from_optimizers(p, optimizers):
    """Helper function to freeze the training of a parameter.

    To freeze a parameter, it must be removed from the optimizer,
    otherwise momentum and weight decay may still be applied.

    Args:
        p: The parameter being frozen.
        optimizers: The optimizers used during training.
    """
    # Force optimizers to be iterable
    if not isinstance(optimizers, (list, tuple)):
        optimizers = [optimizers]

    # Search over params in the optimizers to find and remove the
    # given param. Necessary due to the way params are stored.
    for optimizer in optimizers:
        for group in optimizer.param_groups:
            group['params'] = list(filter(lambda x: id(x) != id(p), group['params']))


def freeze_layers(
    model: Model,
    optimizers: Optimizers,
    current_epoch: int,
    max_epochs: int,
    freeze_start: float,
    freeze_level: float,
    logger: Optional[Logger] = None,
) -> Model:
    """Progressively freeze the layers of the network during training, starting
    with the earlier layers.

    Args:
        model: an instance of the model being trained
        optimizers: the optimizers used during training
        current_epoch: integer specifying the current epoch
        max_epochs: the max number of epochs training will run for
        freeze_start: the fraction of epochs to run before freezing begins
        freeze_level: the maximum fraction of layers to freeze
    """
    # Flatten out the layers
    flat_children = []
    _get_layers(model, flat_children)
    # Determine how many layers to freeze
    freeze_percentage = _freeze_schedule(current_epoch=current_epoch,
                                         max_epochs=max_epochs,
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
    log.info(f'Applied Layer Freezing'
             f' with freeze_start={freeze_start}, '
             f'freeze_level={freeze_level}. '
             f'Froze {freeze_depth} layers in the model which'
             f' equates to {freeze_percentage * 100}% of all layers.')
    if logger is not None:
        logger.metric_epoch({
            'layer_freezing/layers_frozen': freeze_depth,
            'layer_freezing/percentage_frozen': freeze_percentage
        })

    return model


class LayerFreezing(Algorithm):
    """Progressively freeze the layers of the network during training, starting
    with the earlier layers.

    Freezing starts after the fraction of epochs specified by ``freeze_start``
    have run. The fraction of layers frozen increases linearly until it
    reaches ``freeze_level`` at the final epoch.

    This freezing schedule is most similar to
    `FreezeOut <https://arxiv.org/abs/1706.04983>`_ and
    `Freeze Training <https://arxiv.org/abs/1706.05806>`_.

    Runs on ``Event.EPOCH_END``.

    Args:
        freeze_start: the fraction of epochs to run before freezing begins
        freeze_level: the maximum fraction of layers to freeze
    """

    def __init__(self, freeze_start: float = 0.5, freeze_level: float = 1.0):
        self.hparams = LayerFreezingHparams(freeze_start, freeze_level)

    @property
    def find_unused_parameters(self) -> bool:
        """
        Override in order to tell DDP that some parameters will not
        have gradients computed for them after layer freezing is applied.
        """
        return True

    def match(self, event: Event, state: State) -> bool:
        """Run on ``Event.EPOCH_END``."""
        return event == Event.EPOCH_END

    def apply(self, event: Event, state: State, logger: Logger) -> Optional[int]:
        """Freeze layers in the model"""
        optimizers = state.optimizers
        assert optimizers is not None
        state.model = freeze_layers(
            model=state.model,
            optimizers=optimizers,
            current_epoch=state.epoch,
            max_epochs=state.max_epochs,
            freeze_start=self.hparams.freeze_start,
            freeze_level=self.hparams.freeze_level,
            logger=logger,
        )
