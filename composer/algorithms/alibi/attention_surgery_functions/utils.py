# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import logging
import math
import inspect
from operator import attrgetter
from typing import Callable, Dict, Optional, Type

import torch
log = logging.getLogger(__name__)

# Initialize the empty registry that will be filled by using the `register_alibi_replacement_function` decorator.
AlibiReplacementFunction = Callable[[torch.nn.Module, int, int], Optional[torch.nn.Module]]
policy_registry: Dict[Type[torch.nn.Module], AlibiReplacementFunction] = {}


def register_alibi_replacement_function(
        *modules: Type[torch.nn.Module]) -> Callable[[AlibiReplacementFunction], AlibiReplacementFunction]:
    """This decorator builds a registry that maps torch module types to their applicable AlibiReplacementFunction.

    To accommodate the specifics of composer's model surgery, our ALiBi implementation uses a registry to create
    a `Mapping[torch.nn.Module, AlibiReplacementFunction]`, where :func:`AlibiReplacementFunction` is any function that
    takes acts a generic :func:`composer.utils.module_surgery.ReplacementFunction` but with an additional
    `max_sequence_length` keyword argument.

    Implementation files (e.g., `_gpt2.py`) populate the `policy_registry`
    registry by defining instances of `AlibiReplacementFunction` functions and decorating them with
    :func:`register_alibi_replacement_function`.

    Example:

        .. code-block::

            from composer.algorithms.alibi.attention_surgery_functions.utils import register_alibi_replacement_function
            from transformers.models.gpt2.modeling_gpt2 import GPT2Attention

            @register_alibi_replacement_function(GPT2Attention)
            def convert_gpt2_attention(module: torch.nn.Module, index: int, max_sequence_length: int):
                # Builds a function (`convert_attention`) that does model surgery any GPT2Attention modules in the model.

                def convert_attention(module, module_index):
                    # Do surgery (change `module` or generate a new `module` instance to return)
                    # Note that this function can (and often should for ALiBi) depend on `max_sequence_length`

                    # YOUR CODE HERE

                    return module
                return convert_attention

    In the above example, by decorating `build_gpt2_attention_converter` (which is an instance of a `AlibiReplacementFunction`
    function) with `@register_alibi_replacement_function(GPT2Attention)`, the ALiBi algorithm will now apply model surgery to any
    instances of `GPT2Attention` within the model, and will apply surgery on those instances using the `convert_attention` function
    returned by `build_gpt2_attention_converter`.

    Note that `convert_attention` follows the specific signature of a `ReplacementFunction`, which has specific arguments (for additional
    details, see `composer/utils/module_surgery.py`). However, often the model surgery functions used by ALiBi should depend on
    a `max_sequence_length` argument. Since this argument is not used by a `ReplacementFunction`, we make use of these
    `AlibiReplacementFunction` functions so that the exact functions used in model surgery can properly refer to the `max_sequence_length`
    argument, which is not provided until runtime (see `../alibi.py`).
    """
    def _validate_signature(func: Callable):
        # Necessary to enforce that `func` has a valid signature (i.e. is a AlibiReplacementFunction)
        signature = inspect.signature(func)
        parameters = signature.parameters
        if len(parameters) != 3:
            raise ValueError(f'Each alibi surgery function must accept 3 arguments, {func} accepts {len(parameters)}')
        ((_, module_param), (_, index_param), (max_seq_name, max_seq_param)) = parameters.items()
        if module_param.annotation != torch.nn.Module:
            raise TypeError(f'The first argument of alibi surgery function {func} must be of type "torch.nn.Module"')
        if index_param.annotation != int:
            raise TypeError(f'The second argument of alibi surgery function {func} must be of type "int"')
        if max_seq_param.annotation != int:
            raise TypeError(f'The third argument of alibi surgery function {func} must be of type "int"')
        if max_seq_name != 'max_sequence_length':
            raise NameError(f'The third argument of function {func} must be named "max_sequence_length"')

    def _register_module(module: Type[torch.nn.Module], func: Callable) -> None:
        assert issubclass(module, torch.nn.Module)
        if module in policy_registry:
            raise ValueError(f'Module {module.__name__} already has a registered AlibiReplacementFunction.')
        policy_registry[module] = func
        return

    def wrapper(func: AlibiReplacementFunction) -> AlibiReplacementFunction:
        _validate_signature(func)
        for module in modules:
            _register_module(module, func)
        return func

    return wrapper


def zero_and_freeze_expand_position_embeddings(
    module: torch.nn.Module,
    max_sequence_length: int,
    position_embedding_attribute: str,
) -> None:
    """Replaces weights with zero tensor and prevents them from being learned further.

    This is intended to be used specifically for "removing" positional embeddings.
    """
    try:
        pos_embedding_module = attrgetter(position_embedding_attribute)(module)
        old_weight = getattr(pos_embedding_module, 'weight')
        if not isinstance(old_weight, torch.nn.Parameter):
            raise TypeError(f'Module {module._get_name()}, position embedding {position_embedding_attribute}, '
                            f"'weight' attribute must be of type torch.nn.Module")
        new_weight = torch.nn.Parameter(
            torch.zeros((max_sequence_length, old_weight.shape[1]),
                        dtype=old_weight.dtype,
                        layout=old_weight.layout,
                        device=old_weight.device))
        new_weight.requires_grad = False
        setattr(pos_embedding_module, 'weight', new_weight)

        log.info(f' Position embedding expanded to sequence length {max_sequence_length}, zeroed, and frozen')

    except AttributeError:
        log.error(f'Unable to zero and freeze position embeddings. Module '
                  f'{module} may lack attribute {position_embedding_attribute}, or position '
                  f"embeddings may lack attribute 'weight'.")
        raise


def register_alibi(module: torch.nn.Module, n_heads: int, max_token_length: int, causal: bool) -> torch.nn.Module:
    """Adds ALiBi's linear attention biases as a buffer to the module."""
    if causal:  # e.g., for GPT
        # Modified from https://github.com/ofirpress/attention_with_linear_biases/blob/5b327adc6d131e28b40ba58906b30bb469483519/fairseq/models/transformer.py#L742
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
        assert alibi.shape == torch.Size([n_heads, 1, max_token_length])

    else:  # e.g., for BERT
        # Following https://github.com/ofirpress/attention_with_linear_biases/issues/5 (Implementation 1)
        # In the causal case, you can exploit the fact that softmax is invariant to a uniform translation
        # of the logits, which makes the math work out *after* applying causal masking. If no causal masking
        # will be applied, it is necessary to construct the diagonal mask.
        context_position = torch.arange(max_token_length)[:, None]
        memory_position = torch.arange(max_token_length)[None, :]
        relative_position = torch.abs(memory_position - context_position)
        # [n_heads, max_token_length, max_token_length]
        relative_position = relative_position.unsqueeze(0).expand(n_heads, -1, -1)

        slopes = torch.Tensor(_get_alibi_head_slopes(n_heads))
        alibi = slopes.unsqueeze(1).unsqueeze(1) * -relative_position
        # [1, n_heads, max_token_length, max_token_length]
        alibi = alibi.unsqueeze(0)
        assert alibi.shape == torch.Size([1, n_heads, max_token_length, max_token_length])

    module_device = next(module.parameters()).device
    module.register_buffer('alibi', alibi.to(module_device))
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
