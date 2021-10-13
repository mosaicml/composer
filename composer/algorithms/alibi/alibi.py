# Copyright 2021 MosaicML. All Rights Reserved.

from __future__ import annotations

import importlib
import logging
import math
from dataclasses import asdict, dataclass
from operator import attrgetter
from types import MethodType, ModuleType
from typing import Any, Callable, Optional, Union

import torch
import yahp as hp

from composer.algorithms import AlgorithmHparams
from composer.core import Algorithm, Event, Logger, State, surgery

log = logging.getLogger(__name__)


@dataclass
class AlibiHparams(AlgorithmHparams):
    """See :class:`Alibi`"""

    position_embedding_attribute: str = hp.required("attribute name of position embeddings within the model. "
                                                    "For example in HuggingFace's GPT2, the position "
                                                    "embeddings are 'transformer.wpe'")
    attention_module_name: str = hp.required("module/class that will have its self-attention "
                                             "function replaced. For example, in HuggingFace's "
                                             "GPT, the self-attention module is "
                                             "'transformers.models.gpt2.modeling_gpt2.GPT2Attention'")
    attr_to_replace: str = hp.required("model attribute that self-attention function will "
                                       "replace. For example, in HuggingFace's "
                                       "GPT2, the self-attention function is '_attn'")
    alibi_attention: str = hp.required("new self-attention function in which ALiBi is "
                                       "implemented. Used to replace "
                                       "'{attention_module}.{attr_to_replace}'")
    mask_replacement_function: Union[str, None] = hp.optional(
        "function to replace model's attention mask. This is "
        "sometimes necessary for evaluating on sequence "
        " lengths longer than the model was initialized to accommodate.",
        default=None)
    heads_per_layer: Union[int, Optional[None]] = hp.optional(
        'Number of attention heads per layer. If '
        '"None", will attempt to determine from model.config.n_head.',
        default=None)
    max_sequence_length: int = hp.optional('Maximum allowable sequence length', default=8192)
    train_sequence_length_scaling: float = hp.optional(
        'Amount by which to scale training sequence length. One batch of training data '
        'will be reshaped from size (sequence_length, batch) to '
        '(sequence_length*train_sequence_length_scaling, batch/train_sequence_length_scaling)',
        default=0.25)

    def initialize_object(self) -> "Alibi":
        return Alibi(**asdict(self))


def apply_alibi(model: torch.nn.Module, heads_per_layer: int, max_sequence_length: int,
                position_embedding_attribute: str, attention_module: torch.nn.Module, attr_to_replace: str,
                alibi_attention: Callable, mask_replacement_function: Union[Callable, None]) -> None:
    """
    Removes position embeddings and replaces the attention function and attention mask
    according to `AliBi <https://arxiv.org/abs/2108.12409>`_.

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
    """

    zero_and_freeze_expand_position_embeddings(model=model,
                                               attribute=position_embedding_attribute,
                                               new_embedding_length=max_sequence_length)
    log.info(f" Position embedding expanded to sequence " f"length {max_sequence_length}, zeroed, and frozen")

    def convert_attention(module: torch.nn.Module, module_index: int = None):
        module = register_alibi(module=module, n_heads=heads_per_layer, max_token_length=max_sequence_length)
        setattr(module, attr_to_replace, MethodType(alibi_attention, module))
        if mask_replacement_function:
            module = mask_replacement_function(module, max_sequence_length)
        return module

    transforms = {attention_module: convert_attention}
    replaced_pairs = surgery.replace_module_classes(model, transforms)  # type: ignore

    count = len(replaced_pairs)
    log.info(f" {count} instances of ALiBi added")


class Alibi(Algorithm):
    """
    `AliBi <https://arxiv.org/abs/2108.12409>`_ (Attention with Linear Biases)
    dispenses with position embeddings and instead directly biases attention
    matrices such that nearby tokens attend to one another more strongly.

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

    def __init__(self, position_embedding_attribute: str, attention_module_name: str, attr_to_replace: str,
                 alibi_attention: str, mask_replacement_function: str, heads_per_layer: int, max_sequence_length: int,
                 train_sequence_length_scaling: float) -> None:

        self.hparams = AlibiHparams(position_embedding_attribute=position_embedding_attribute,
                                    attention_module_name=attention_module_name,
                                    attr_to_replace=attr_to_replace,
                                    alibi_attention=alibi_attention,
                                    mask_replacement_function=mask_replacement_function,
                                    heads_per_layer=heads_per_layer,
                                    max_sequence_length=max_sequence_length,
                                    train_sequence_length_scaling=train_sequence_length_scaling)

    def match(self, event: Event, state: State) -> bool:
        """ Runs on Event.INIT
        """
        return event in (Event.INIT, Event.AFTER_DATALOADER)

    def apply(self, event: Event, state: State, logger: Logger) -> Optional[int]:
        """ Replace model's existing attention mechanism with AliBi
        """

        if event == Event.INIT:
            assert state.model is not None

            if "heads_per_layer" not in asdict(self.hparams).keys() or \
            not self.hparams.heads_per_layer:
                try:
                    self.hparams.heads_per_layer = state.model.config.n_head  # type: ignore
                except AttributeError:
                    log.exception("alibi.heads_per_layer not provided, and unable to "
                                  "determine number of heads from model.config.n_head."
                                  " Please provide alibi. heads_per_layer.")

            apply_alibi(
                state.model,
                heads_per_layer=self.hparams.heads_per_layer,  # type: ignore
                max_sequence_length=self.hparams.max_sequence_length,
                position_embedding_attribute=self.hparams.position_embedding_attribute,
                attr_to_replace=self.hparams.attr_to_replace,
                # Access method from string
                attention_module=lazy_import(self.hparams.attention_module_name),
                # Access method from string
                alibi_attention=lazy_import(self.hparams.alibi_attention),
                # Access method from string
                mask_replacement_function=lazy_import(self.hparams.mask_replacement_function))

        elif event == Event.AFTER_DATALOADER:
            # Change sequence length by reshaping data
            if not self.hparams.train_sequence_length_scaling == 1 and \
            hasattr(state, "batch") and isinstance(state.batch, dict):
                sequence_scaling = self.hparams.train_sequence_length_scaling
                for k, v in state.batch.items():
                    batch_len, sequence_len = v.shape[0], v.shape[1]
                    state.batch[k] = v.reshape(int(batch_len / sequence_scaling), int(sequence_len * sequence_scaling))


def zero_and_freeze_expand_position_embeddings(model: torch.nn.Module, new_embedding_length: int, attribute: str):
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


def register_alibi(module: torch.nn.Module, n_heads: int, max_token_length: int):
    # Modified from https://github.com/ofirpress/attention_with_linear_biases/blob/master/fairseq/models/transformer.py#L742
    slopes = torch.Tensor(get_alibi_head_slopes(n_heads))
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


def get_alibi_head_slopes(n_heads: int):

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
        return get_slopes_power_of_2(closest_power_of_2) + get_alibi_head_slopes(
            2 * closest_power_of_2)[0::2][:n_heads - closest_power_of_2]


def lazy_import(name: Union[str, None]) -> Any[Callable, ModuleType, None]:
    if not name:
        return None
    components = name.split('.')
    try:
        mod = importlib.import_module(components[0])
    except (ValueError, ModuleNotFoundError):
        log.exception(f"Module {components[0]} not found when attempting "
                      f"to import {name}. Please confirm the name and "
                      f"module path you're attempting to import.")
    try:
        mod = attrgetter('.'.join(components[1:]))(mod)  # type: ignore
    except (ValueError, AttributeError):
        log.exception(f"Unable to import {name}. "
                      f"Please confirm the name and module "
                      f" path you're attempting to import.")
    return mod  # type: ignore
