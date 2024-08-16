# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

# Copyright 2022 MosaicML. All Rights Reserved.

from __future__ import annotations

import logging
import warnings
from typing import Callable, Optional, Sequence, Union

import torch

from composer.models.huggingface import HuggingFaceModel

try:
    from transformers import BertPreTrainedModel
    from transformers.models.bert.modeling_bert import BertIntermediate, BertOutput
    IS_TRANSFORMERS_INSTALLED = True
except ImportError as e:
    IS_TRANSFORMERS_INSTALLED = False

from composer.algorithms.gated_linear_units.gated_linear_unit_layers import BERTGatedFFOutput
from composer.algorithms.warnings import NoEffectWarning
from composer.core import Algorithm, Event, State
from composer.loggers import Logger
from composer.utils import MissingConditionalImportError, module_surgery

log = logging.getLogger(__name__)


def from_BertOutput(
    layer: torch.nn.Module,
    module_index: int,
    act_fn: Callable[[torch.Tensor], torch.Tensor],
    gated_layer_bias: bool = False,
    non_gated_layer_bias: bool = False,
) -> BERTGatedFFOutput:
    """Defines a replacement policy from a :class:`transformers.models.bert.modeling_bert.BertOutput` to a :class:`composer.algorithms.gated_linear_units.gated_linear_unit_layers.BERTGatedFFOutput`"""
    assert isinstance(
        layer,
        BertOutput,  # pyright: ignore[reportUnboundVariable]
    ), 'The replacement policy requires an instance of transformers.models.bert.modeling_bert.BertOutput for the necessary fields to be defined.'
    return BERTGatedFFOutput(
        d_embed=layer.dense.out_features,  #type: ignore dense.out_features member of BertOutput
        d_ff=layer.dense.in_features,  #type: ignore dense.in_features member of BertOutput
        dropout_rate=layer.dropout.p,  #type: ignore dropout.p member of BertOutput
        act_fn=act_fn,
        layernorm_eps=layer.LayerNorm.eps,  #type: ignore LayerNorm.eps member of BertOutput
        gated_layer_bias=gated_layer_bias,
        non_gated_layer_bias=non_gated_layer_bias,
    )


def from_BertIntermediate(layer: torch.nn.Module, module_index: int) -> torch.nn.Identity:
    """
    Defines a replacement policy from a :class:`transformers.models.bert.modeling_bert.BertIntermediate` to a :class:`torch.nn.Identity`
    The identity effectively acts as no-op.
    """
    return torch.nn.Identity()


def apply_gated_linear_units(
    model: torch.nn.Module,
    optimizers: Union[torch.optim.Optimizer, Sequence[torch.optim.Optimizer]],
    act_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    gated_layer_bias: bool = False,
    non_gated_layer_bias: bool = False,
) -> None:
    """
    Replaces the Linear layers in the feed-forward network with `Gated Linear Units <https://arxiv.org/abs/2002.05202>`_.

    Args:
        model (`torch.nn.Module`): The model to modify in-place.
        optimizers (`torch.optim.Optimizer` | Sequence[`torch.optim.Optimizer`], optional):
            Existing optimizers bound to ``model.parameters()``. All optimizers that have already been
            constructed with ``model.parameters()`` must be specified here so that
            they will optimize the correct parameters.

            If the optimizer(s) are constructed after calling this function,
            then it is safe to omit this parameter. These optimizers will see the correct
            model parameters.
        act_fn (Callable[torch.Tensor, torch.Tensor], optional): Optionally, the activation function to use. If ``None``, the algorithm will
            use the existing activation function in the model.
        gated_layer_bias (bool, optional): Whether to use biases in the linear layers within the GLU. Default: ``False``.
        non_gated_layer_bias (bool, optional): Whether to use biases in the linear layers within the GLU. Default: ``False``.
    """
    if not IS_TRANSFORMERS_INSTALLED:
        raise MissingConditionalImportError(extra_deps_group='nlp', conda_package='transformers')

    unwrapped_model = model.model if isinstance(model, HuggingFaceModel) else model

    # ensure that the model is an instance of a Hugging Face BertPreTrainedModel class, since our replacement policy is only defined for BERTs
    if not isinstance(unwrapped_model, BertPreTrainedModel):  # pyright: ignore[reportUnboundVariable]
        raise TypeError(
            'Gated Linear Units only has a surgery policy defined for subclasses of transformers.BertPreTrainedModel',
        )

    # Early exit if nothing to replace
    if module_surgery.count_module_instances(
        module=model,
        module_class=BertIntermediate,
    ) == 0:  # pyright: ignore[reportUnboundVariable]
        return

    if act_fn is None:
        intermediate_modules = {
            module for module in model.modules()
            if isinstance(module, BertIntermediate)  # pyright: ignore[reportUnboundVariable]
        }  # pyright: ignore[reportUnboundVariable]
        if len(intermediate_modules) == 0:
            warnings.warn(
                NoEffectWarning(
                    'No instances of BertIntermediate were found so Gated Linear Units will be skipped '
                    'as no modules can be replaced. This is likely because Gated Linear Units has already '
                    'been applied to this model.',
                ),
            )

        # get the activation functions used
        act_fns = {module.intermediate_act_fn for module in intermediate_modules}
        num_act_fns = len({type(act_fn) for act_fn in act_fns})
        if num_act_fns == 0:
            raise ValueError(
                'Tried to get the activation function from the model, but none were found. '
                'Please specify `act_fn` manually to use Gated Linear Units.',
            )
        elif num_act_fns > 1:
            raise ValueError(
                'Tried to get the activation function from the model, but multiple different '
                'functions are used. This is currently unsupported with Gated Linear Units. '
                'Please either use one activation function in BertIntermediate modules or '
                'specify `act_fn` to manually override activation functions.',
            )

        # since our set is of 1, let's extract the activation function
        act_fn = next(iter(act_fns))  # type: ignore will fail below if None

        if act_fn is None:
            raise ValueError(
                'Found activation function was None. If this is an error, please manually specify `act_fn`.',
            )

    # now that we know the act fn, bind a few parameters of the replacement function
    def from_bound_BertOutput(layer: torch.nn.Module, module_index: int) -> BERTGatedFFOutput:
        return from_BertOutput(
            layer=layer,
            module_index=module_index,
            act_fn=act_fn,
            gated_layer_bias=gated_layer_bias,
            non_gated_layer_bias=non_gated_layer_bias,
        )

    # prepare the replacement policy and perform replacement
    policy: dict[type[torch.nn.Module], module_surgery.ReplacementFunction] = {
        BertIntermediate: from_BertIntermediate,  # pyright: ignore[reportUnboundVariable]
        BertOutput: from_bound_BertOutput,  # pyright: ignore[reportUnboundVariable]
    }
    replaced_instances = module_surgery.replace_module_classes(module=model, optimizers=optimizers, policies=policy)
    if len(replaced_instances) == 0:
        warnings.warn(
            NoEffectWarning('No instances of BertIntermediate and BertOutput were found so no modules were replaced.'),
        )
    log.info(
        f'Successfully replaced {len(replaced_instances)} of BertIntermediate and BertOutput with a GatedLinearUnit.',
    )


class GatedLinearUnits(Algorithm):
    """Replaces all instances of Linear layers in the feed-forward subnetwork with a `Gated Linear Unit <https://arxiv.org/abs/2002.05202>`_.
    The Gated Linear Units provide a more expressive form for the same number of parameters, and a slight degradation to throughput.

    Runs on :attr:`.Event.INIT`, so it can swap the Linear layers in the FFN for GLUs before the model is DDP wrapped.

    Args:
        act_fn (Callable[[torch.Tensor], torch.Tensor], optional): Optionally, the activation function to use. If ``None``, the algorithm will
            use the existing activation function in the model.
        gated_layer_bias (bool, optional): Whether to use biases in the linear layers within the GLU. Default: ``False``.
        non_gated_layer_bias (bool, optional): Whether to use biases in the linear layers within the GLU. Default: ``False``.

    Example:
        .. testsetup::

           from tests.common.models import configure_tiny_bert_hf_model
           from tests.common.datasets import dummy_bert_lm_dataloader

           model, train_dataloader = configure_tiny_bert_hf_model(), dummy_bert_lm_dataloader()

        .. testcode::

           from composer.algorithms import GatedLinearUnits

           algorithm = GatedLinearUnits()
           trainer = Trainer(
               model=model,
               train_dataloader=train_dataloader,
               max_duration="1ep",
               algorithms=[algorithm],
               optimizers=[optimizer]
           )
    """

    def __init__(
        self,
        act_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        gated_layer_bias: bool = False,
        non_gated_layer_bias: bool = False,
    ):
        if not IS_TRANSFORMERS_INSTALLED:
            raise MissingConditionalImportError(extra_deps_group='nlp', conda_package='transformers')
        self.act_fn = act_fn
        self.gated_layer_bias = gated_layer_bias
        self.non_gated_layer_bias = non_gated_layer_bias

    def __repr__(self) -> str:
        act_fn = 'act_fn' if self.act_fn else None
        return f'{self.__class__.__name__}(act_fn={act_fn},gated_layer_bias={self.gated_layer_bias},non_gated_layer_bias={self.non_gated_layer_bias})'

    @staticmethod
    def required_on_load() -> bool:
        return True

    def match(self, event: Event, state: State) -> bool:
        del state  # unused
        return event == Event.INIT

    def apply(self, event: Event, state: State, logger: Logger) -> Optional[int]:
        del event, logger  # unused
        apply_gated_linear_units(
            model=state.model,
            optimizers=state.optimizers,
            act_fn=self.act_fn,
            gated_layer_bias=self.gated_layer_bias,
            non_gated_layer_bias=self.non_gated_layer_bias,
        )
