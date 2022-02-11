# Copyright 2021 MosaicML. All Rights Reserved.

from __future__ import annotations

import importlib
import logging
import math
from operator import attrgetter
from types import MethodType, ModuleType
from typing import Any, Callable, Optional, Type, Union, cast

import torch

from composer.core import Algorithm, Event, Logger, State
from composer.core.types import Optimizers
from composer.utils import module_surgery

log = logging.getLogger(__name__)


def apply_alibi(
    model: torch.nn.Module,
    heads_per_layer: int,
    max_sequence_length: int,
    position_embedding_attribute: str,
    attention_module: Type[torch.nn.Module],
    attr_to_replace: str,
    alibi_attention: Callable,
    mask_replacement_function: Union[Callable, None],
    optimizers: Optional[Optimizers] = None,
) -> None:
    """Removes position embeddings and replaces the attention function and attention mask according to `AliBi.

    <https://arxiv.org/abs/2108.12409>`_.

    Args:
        model: model to transform
        heads_per_layer: number of attention heads per layer
        max_sequence_length: maximum sequence length that the
            model will be able to accept without returning an error
        position_embedding_attribute: attribute for position
            embeddings. For example in HuggingFace's GPT2, the
            position embeddings are "transformer.wpe".
        attention_module: module/class that will have its
            self-attention function replaced. For example, in
            HuggingFace's GPT, the self-attention module is
            transformers.models.gpt2.modeling_gpt2.GPT2Attention.
        attr_to_replace: attribute that self-attention function will
            replace. For example, in HuggingFace's GPT2, the
            self-attention function is "_attn".
        alibi_attention: new self-attention function in which
            ALiBi is implemented. Used to replace
            "{attention_module}.{attr_to_replace}".
        mask_replacement_function: function to replace model's
            attention mask. This is sometimes necessary for evaluating
            on sequence lengths longer than the model was initialized to
            accommodate.
        optimizers (Optimizers, optional): Existing optimizers bound to ``model.parameters()``.
            All optimizers that have already been constructed with,
            ``model.parameters()`` must be specified here so they will optimize
            the correct parameters.

            If the optimizer(s) are constructed *after* calling this function,
            then it is safe to omit this parameter. These optimizers will see the correct
            model parameters.
    """

    _zero_and_freeze_expand_position_embeddings(model=model,
                                                attribute=position_embedding_attribute,
                                                new_embedding_length=max_sequence_length)
    log.info(f" Position embedding expanded to sequence length {max_sequence_length}, zeroed, and frozen")

    def convert_attention(module: torch.nn.Module, module_index: Optional[int] = None):
        del module_index  # unused
        module = _register_alibi(module=module, n_heads=heads_per_layer, max_token_length=max_sequence_length)
        setattr(module, attr_to_replace, MethodType(alibi_attention, module))
        if mask_replacement_function:
            module = mask_replacement_function(module, max_sequence_length)
        return module

    replaced_pairs = module_surgery.replace_module_classes(model,
                                                           optimizers=optimizers,
                                                           policies={attention_module: convert_attention})

    count = len(replaced_pairs)
    log.info(f" {count} instances of ALiBi added")


class Alibi(Algorithm):
    """`ALiBi <https://arxiv.org/abs/2108.12409>`_ (Attention with Linear Biases) dispenses with position embeddings and
    instead directly biases attention matrices such that nearby tokens attend to one another more strongly.

    ALiBi yields excellent extrapolation to unseen sequence lengths
    compared to other position embedding schemes. We leverage this
    extrapolation capability by training with shorter sequence lengths,
    which reduces the memory and computation load.

    This algorithm modifies the model and runs on Event.INIT. This algorithm
    should be applied before the model has been moved to accelerators.

    Args:
        heads_per_layer: number of attention heads per layer
        max_sequence_length: maximum sequence length that the
            model will be able to accept without returning an error
        position_embedding_attribute: attribute for position
            embeddings. For example in HuggingFace's GPT2, the
            position embeddings are "transformer.wpe".
        attention_module_name: module/class that will have
            its self-attention function replaced. For example,
            in HuggingFace's GPT, the self-attention module is
            "transformers.models.gpt2.modeling_gpt2.GPT2Attention".
        attr_to_replace: attribute that self-attention function will
            replace. For example, in HuggingFace's GPT2, the
            self-attention function is "_attn".
        alibi_attention: Path to new self-attention function in which ALiBi is
            implemented. Used to replace "{attention_module}.{attr_to_replace}".
        mask_replacement_function: Path to function to replace model's
            attention mask. This is sometimes necessary for evaluating on
            sequence lengths longer than the model was initialized to
            accommodate.
        train_sequence_length_scaling: Amount by which to scale
            training sequence length. One batch of training data will be
            reshaped from size (sequence_length, batch) to
            (sequence_length*sequence_length_fraction, batch/sequence_length_fraction).
    """

    def __init__(self,
                 position_embedding_attribute: str,
                 attention_module_name: str,
                 attr_to_replace: str,
                 alibi_attention: str,
                 mask_replacement_function: Optional[str] = None,
                 heads_per_layer: Optional[int] = None,
                 max_sequence_length: int = 8192,
                 train_sequence_length_scaling: float = 0.25) -> None:

        self.position_embedding_attribute = position_embedding_attribute
        self.attention_module_name = attention_module_name
        self.attr_to_replace = attr_to_replace
        self.alibi_attention = alibi_attention
        self.mask_replacement_function = mask_replacement_function
        self.heads_per_layer = heads_per_layer
        self.max_sequence_length = max_sequence_length
        self.train_sequence_length_scaling = train_sequence_length_scaling
        self._applied = False

    def match(self, event: Event, state: State) -> bool:
        """Runs on Event.INIT."""
        return (event == Event.INIT and not self._applied) or event == Event.AFTER_DATALOADER

    def apply(self, event: Event, state: State, logger: Logger) -> Optional[int]:
        """Replace model's existing attention mechanism with AliBi."""

        if event == Event.INIT:

            if self.heads_per_layer is None:
                try:
                    self.heads_per_layer = state.model.config.n_head  # type: ignore
                except AttributeError:
                    log.exception("alibi.heads_per_layer not provided, and unable to "
                                  "determine number of heads from model.config.n_head."
                                  " Please provide alibi. heads_per_layer.")

            apply_alibi(
                state.model,
                optimizers=state.optimizers,
                heads_per_layer=cast(int, self.heads_per_layer),
                max_sequence_length=self.max_sequence_length,
                position_embedding_attribute=self.position_embedding_attribute,
                attr_to_replace=self.attr_to_replace,
                # Access method from string
                attention_module=_lazy_import(self.attention_module_name),
                # Access method from string
                alibi_attention=_lazy_import(self.alibi_attention),
                # Access method from string
                mask_replacement_function=_lazy_import(self.mask_replacement_function))

            self._applied = True

        elif event == Event.AFTER_DATALOADER:
            # Change sequence length by reshaping data
            if not self.train_sequence_length_scaling == 1 and \
            hasattr(state, "batch") and isinstance(state.batch, dict):
                sequence_scaling = self.train_sequence_length_scaling
                for k, v in state.batch.items():
                    batch_len, sequence_len = v.shape[0], v.shape[1]
                    state.batch[k] = v.reshape(int(batch_len / sequence_scaling), int(sequence_len * sequence_scaling))


def _zero_and_freeze_expand_position_embeddings(model: torch.nn.Module, new_embedding_length: int, attribute: str):
    try:
        pos_embedding_module = attrgetter(attribute)(model)
        old_weight = getattr(pos_embedding_module, "weight")
        new_weight = torch.nn.Parameter(
            torch.zeros((new_embedding_length, old_weight.shape[1]),
                        dtype=old_weight.dtype,
                        layout=old_weight.layout,
                        device=old_weight.device))
        new_weight.requires_grad = False
        setattr(pos_embedding_module, "weight", new_weight)
    except AttributeError:
        log.error(f"Unable to zero and freeze position embeddings. Model "
                  f"{model} may lack attribute {attribute}, or position "
                  f"embeddings may lack attribute 'weight'.")


def _register_alibi(module: torch.nn.Module, n_heads: int, max_token_length: int):
    # Modified from https://github.com/ofirpress/attention_with_linear_biases/blob/master/fairseq/models/transformer.py#L742
    slopes = torch.Tensor(_get_alibi_head_slopes(n_heads))
    # In the next line, the part after the * is what constructs the diagonal matrix
    # (right matrix in Figure 3 in the paper).
    # If you run it you'll see that it doesn't exactly print out the same matrix as we
    # have in Figure 3, but one where all rows are identical.
    # This works because the softmax operation is invariant to translation, and our bias
    # functions are always linear.
    alibi = slopes.unsqueeze(1).unsqueeze(1) * \
        torch.arange(max_token_length). \
        unsqueeze(0).unsqueeze(0).expand(n_heads, -1, -1)
    module.register_buffer("alibi", alibi)
    return module


def _get_alibi_head_slopes(n_heads: int):

    def get_slopes_power_of_2(n_heads):
        start = (2**(-2**-(math.log2(n_heads) - 3)))
        ratio = start
        return [start * ratio**i for i in range(n_heads)]

    # In the paper, they only train models that have 2^a heads for some a. This function
    # has some good properties that only occur when the input is a power of 2. To
    # maintain that even when the number of heads is not a power of 2, we use a
    # workaround.
    if math.log2(n_heads).is_integer():
        return get_slopes_power_of_2(n_heads)
    else:
        closest_power_of_2 = 2**math.floor(math.log2(n_heads))
        return get_slopes_power_of_2(closest_power_of_2) + _get_alibi_head_slopes(
            2 * closest_power_of_2)[0::2][:n_heads - closest_power_of_2]


def _lazy_import(name: Optional[str]) -> Any[Callable, ModuleType, None]:
    if not name:
        return None
    components = name.split('.')
    try:
        mod = importlib.import_module(components[0])
    except (ValueError, ModuleNotFoundError):
        log.exception(f"Module {components[0]} not found when attempting "
                      f"to import {name}. Please confirm the name and "
                      f"module path you're attempting to import.")
        raise
    try:
        mod = attrgetter('.'.join(components[1:]))(mod)
    except (ValueError, AttributeError):
        log.exception(f"Unable to import {name}. "
                      f"Please confirm the name and module "
                      f" path you're attempting to import.")
        raise
    return mod
