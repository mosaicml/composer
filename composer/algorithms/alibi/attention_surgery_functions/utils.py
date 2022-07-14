# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import math
import torch
import logging
from typing import Dict, TypeVar, Callable
from composer.utils.module_surgery import ReplacementFunction
from operator import attrgetter

log = logging.getLogger(__name__)


# Initialize the empty registry that will be filled by using the `register_surgery_function_builder` decorator.
SurgeryFunctionBuilder = TypeVar('SurgeryFunctionBuilder', bound=Callable[[int], ReplacementFunction])
replacement_policy_mapping_builder: Dict[torch.nn.Module, SurgeryFunctionBuilder] = dict()

def register_surgery_function_builder(*modules: torch.nn.Module) -> Callable[[SurgeryFunctionBuilder], SurgeryFunctionBuilder]:
    """This decorator builds a registry that maps torch modules to their applicable SurgeryFunctionBuilder.

    To accommodate the specifics of composer's model surgery, the register is used to create
    a `Mapping[torch.nn.Module, SurgeryFunctionBuilder]`, where :func:`SurgeryFunctionBuilder` is any function that
    takes a `max_sequence_length` argument and returns a :func:`composer.utils.model_surgery.ReplacementFunction`.

    At runtime (see `../alibi.py`), ALiBi gets the model surgery policy mapping 
    `Mapping[torch.nn.Module, ReplacementFunction]` by replacing each `SurgeryFunctionBuilder` with the 
    `ReplacementFunction` it returns, then using the resulting `policies` mapping to perform
    model surgery:

    .. code-block:: python

        from composer.algorithms.alibi.attention_surgery_functions import replacement_policy_mapping_builder
        from composer.utils import model_surgery

        policies = {
            module: surgery_function_builder(max_sequence_length)
            for module, surgery_function_builder in replacement_policy_mapping_builder.items()
        }

        module_surgery.replace_module_classes(model, policies=policies)


    Implementation files (e.g., `../attention_surgery_functions/_gpt2.py`) populate the `replacement_policy_mapping_builder`
    registry by defining instances of `SurgeryFunctionBuilder` functions and decorating them with
    :func:`register_surgery_function_builder`. For example:

    .. code-block:: python
        from composer.algorithms.alibi.attention_surgery_functions.utils import register_surgery_function_builder
        from transformers.models.gpt2.modeling_gpt2 import GPT2Attention

        @register_surgery_function_builder(GPT2Attention)
        def build_gpt2_attention_converter(max_sequence_length: int):
            # Builds a function (`convert_attention`) that does model surgery any GPT2Attention modules in the model.
            
            def convert_attention(module: GPT2Attention, module_index: Optional[int] = None):
                # Do surgery (change `module` or generate a new `module` instance to return)
                # Note that this function can (and often should for ALiBi) depend on `max_sequence_length`
                ...
                return module
            return convert_attention

    In the above example, by decorating `build_gpt2_attention_converter` (which is an instances of a `SurgeryFunctionBuilder`
    function) with `@register_surgery_function_builder(GPT2Attention)`, the ALiBi algorithm will now apply model surgery to any
    instances of `GPT2Attention` within the model, and will apply surgery on those instances using the `convert_attention` function
    returned by `build_gpt2_attention_converter`.

    Note that `convert_attention` follows the specific signature of a `ReplacementFunction`, which has specific arguments. However,
    often the model surgery functions used by ALiBi should depend on a `max_sequence_length` argument. Since this argument is not used
    by a `ReplacementFunction`, we make use of these `SurgeryFunctionBuilder` functions so that the exact functions used in model surgery
    can properly refer to the `max_sequence_length` argument, which is not provided until runtime (see `../alibi.py`).
    """
    def _register_module(module: torch.nn.Module, func: Callable) -> None:
        if module in replacement_policy_mapping_builder:
            raise ValueError(f"Module {module.__name__} already has a registered SurgeryFunctionBuilder.")
        replacement_policy_mapping_builder[module] = func
        return

    def wrapper(func: SurgeryFunctionBuilder) -> SurgeryFunctionBuilder:
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
            raise TypeError(
                f"Module {module._get_name()}, position embedding {position_embedding_attribute}, "
                f"'weight' attribute must be of type torch.nn.Module"
            )
        new_weight = torch.nn.Parameter(
            torch.zeros((max_sequence_length, old_weight.shape[1]),
                        dtype=old_weight.dtype,
                        layout=old_weight.layout,
                        device=old_weight.device))
        new_weight.requires_grad = False
        setattr(pos_embedding_module, 'weight', new_weight)

        log.info(f' Position embedding expanded to sequence length {max_sequence_length}, zeroed, and frozen')

    except AttributeError:
        # TODO: This should perhaps result in an exception rather than a warning
        log.error(f'Unable to zero and freeze position embeddings. Module '
                  f'{module} may lack attribute {position_embedding_attribute}, or position '
                  f"embeddings may lack attribute 'weight'.")


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