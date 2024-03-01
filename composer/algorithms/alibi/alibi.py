# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Core ALiBi classes and functions."""

from __future__ import annotations

import logging
from typing import Optional, Sequence, Union

import torch
from torch.optim import Optimizer

from composer.core import Algorithm, Event, State
from composer.loggers import Logger
from composer.utils import MissingConditionalImportError, module_surgery

log = logging.getLogger(__name__)

__all__ = ['Alibi', 'apply_alibi']


def apply_alibi(
    model: torch.nn.Module,
    max_sequence_length: int,
    optimizers: Optional[Union[Optimizer, Sequence[Optimizer]]] = None,
) -> None:
    """Removes position embeddings and replaces the attention function and attention mask
    as per :class:`.Alibi`. Note that the majority of the training speed-up from using ALiBi
    comes from being able to train on shorter sequence lengths; this function does not scale
    the training sequence length as :class:`.Alibi` does, so little speedup will be
    observed from using it alone. See the :doc:`Method Card </method_cards/alibi>` for
    more details. This function should be called after the model is instantiated and
    before training begins.

    Example:

    .. code-block:: python

        import composer.functional as cf

        cf.apply_alibi(
            model=model,
            max_sequence_length=512
        )

    Args:
        model (torch.nn.Module): Model to transform.
        max_sequence_length (int): Maximum sequence length that the
            model will be able to accept. Internally, the transformations applied by alibi
            change sequence-shaped tensors to handle sequences up to ``max_sequence_length``.
            Depending on ``max_sequence_length`` and ``model`` these changes could increase
            or decrease the model's maximum sequence length.

            At minimum, ``max_sequence_length`` should be set to the sequence length used
            during training. However, if evaluating on sequence lengths longer than those
            used in training, ``max_sequence_length`` should be set accordingly.

            Note that larger ``max_sequence_length`` means a larger memory footprint of
            the model. So, it is best to set this parameter equal the longest
            sequence length that will be seen during training and/or evaluation.
        optimizers (torch.optim.Optimizer | Sequence[torch.optim.Optimizer], optional):
            Existing optimizers bound to ``model.parameters()``. All optimizers that have already been
            constructed with ``model.parameters()`` must be specified here so
            they will optimize the correct parameters.

            If the optimizer(s) are constructed *after* calling this function,
            then it is safe to omit this parameter. These optimizers will see the correct
            model parameters.
    """
    try:
        from composer.algorithms.alibi.attention_surgery_functions import policy_registry
    except ImportError as e:
        raise MissingConditionalImportError(extra_deps_group='nlp', conda_package='transformers') from e

    # To use model surgery utilities, we need to define a policy of type
    # Mapping[Type[torch.nn.Module], ReplacementFunction], where ReplacementFunction is
    # Callable[[torch.nn.Module, Optional[int]], Optional[torch.nn.Module]].
    #
    # This mapping is built by the source code in `./attention_surgery_functions/` but
    # needs to be completed here by "freezing" alibi-specific arguments.
    #
    # For additional details, see `./attention_surgery_functions/utils.py`.
    def as_replacement_function(surgery_function):

        def replacement_function(module: torch.nn.Module, module_index: int):
            return surgery_function(module, module_index, max_sequence_length=max_sequence_length)

        return replacement_function

    # Wrap each alibi_surgery_function as a ReplacementFunction by "freezing" `max_sequence_length`
    policies = {
        module_class: as_replacement_function(alibi_surgery_function)
        for module_class, alibi_surgery_function in policy_registry.items()
    }

    # Note: `policies` defines replacements for _all_ the modules registered in `policy_registry`,
    # meaning that some replacements may be irrelevant for `model`.
    # Conversely, attention modules within `model` may be ignored if they are not registered by the
    # implementations within `./attention_surgery_functions/`.
    replaced_pairs = module_surgery.replace_module_classes(model, optimizers=optimizers, policies=policies)

    count = len(replaced_pairs)
    if count == 0:
        supported_modules = ''.join(sorted(['\n\t' + c.__module__ + '.' + c.__name__ for c in policy_registry.keys()]))
        log.warning(
            f'ALiBi had no effect on the model! Support for ALiBi surgery '
            f'is currently limited to the following classes: {supported_modules}',
        )
    else:
        log.info(f' {count} instances of ALiBi added')


class Alibi(Algorithm):
    """ALiBi (Attention with Linear Biases; `Press et al, 2021  <https://arxiv.org/abs/2108.12409>`_) dispenses with
    position embeddings and instead directly biases attention matrices such that nearby tokens attend to one another
    more strongly.

    ALiBi yields excellent extrapolation to unseen sequence lengths
    compared to other position embedding schemes. We leverage this
    extrapolation capability by training with shorter sequence lengths,
    which reduces the memory and computation load.

    This algorithm runs on :attr:`.Event.INIT` to modify the model
    before the model has been moved to accelerators. It also runs on
    :attr:`.Event.AFTER_DATALOADER` to modify the shape of a batch of
    data after the model and data have been moved to accelerators.

    See the :doc:`Method Card </method_cards/alibi>` for more details.

    Example:

    .. code-block::

        from composer.algorithms import Alibi
        from composer.trainer import Trainer

        alibi = Alibi(
            max_sequence_length=512,
            train_sequence_length_scaling=0.25,
        )

        trainer = Trainer(
            model=model,
            train_dataloader=train_dataloader,
            max_duration="1ep",
            algorithms=[alibi]
        )

    Args:
        max_sequence_length (int): Maximum sequence length that the
            model will be able to accept. This is sometimes necessary for evaluating
            on sequence lengths longer than the model was initialized to
            accommodate.
        train_sequence_length_scaling (float, optional): Amount by which to scale
            training sequence length. One batch of training data will be
            reshaped from shape :math:`(sequence\\_length, batch)` to
            :math:`(sequence\\_length \\times train\\_sequence\\_length\\_scaling,
            \\frac{batch}{train\\_sequence\\_length\\_scaling})`. Default: ``0.25``.
    """

    def __init__(self, max_sequence_length: int, train_sequence_length_scaling: float = 0.25) -> None:
        self.max_sequence_length = max_sequence_length
        self.train_sequence_length_scaling = train_sequence_length_scaling
        self._applied = False

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(max_sequence_length={self.max_sequence_length},train_sequence_length_scaling={self.train_sequence_length_scaling})'

    @staticmethod
    def required_on_load() -> bool:
        return True

    def match(self, event: Event, state: State) -> bool:
        return (event == Event.INIT and not self._applied) or event == Event.AFTER_DATALOADER

    def apply(self, event: Event, state: State, logger: Logger) -> Optional[int]:
        if event == Event.INIT:
            apply_alibi(
                state.model,
                optimizers=state.optimizers,
                max_sequence_length=self.max_sequence_length,
            )

            self._applied = True

        elif event == Event.AFTER_DATALOADER:
            # Change sequence length by reshaping data
            if not self.train_sequence_length_scaling == 1 and \
            hasattr(state, 'batch') and isinstance(state.batch, dict):
                sequence_scaling = self.train_sequence_length_scaling
                for k, v in state.batch.items():
                    batch_len, sequence_len = v.shape[0], v.shape[1]
                    state.batch[k] = v.reshape(int(batch_len / sequence_scaling), int(sequence_len * sequence_scaling))
