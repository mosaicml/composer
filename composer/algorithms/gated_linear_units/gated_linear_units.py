# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

# Copyright 2022 MosaicML. All Rights Reserved.

from __future__ import annotations

import logging
import warnings
from typing import Callable, Dict, Optional, Sequence, Type, Union

import torch

from composer.models.huggingface import HuggingFaceModel

try:
    from transformers import BertForMaskedLM, BertForSequenceClassification
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


def from_BertOutput(layer: torch.nn.Module,
                    module_index: int,
                    act_fn: Callable[[torch.Tensor], torch.Tensor],
                    gated_layer_bias: bool = False,
                    non_gated_layer_bias: bool = False) -> BERTGatedFFOutput:
    """Defines a replacement policy from a :class:`transformers.models.bert.modeling_bert.BertOutput` to a :class:`composer.algorithms.gated_linear_units.gated_linear_unit_layers.BERTGatedFFOutput`"""
    assert isinstance(
        layer, BertOutput
    ), 'The replacement policy requires an instance of transformers.models.bert.modeling_bert.BertOutput for the necessary fields to be defined.'
    return BERTGatedFFOutput(d_embed=layer.dense.out_features,
                             d_ff=layer.dense.in_features,
                             dropout_rate=layer.dropout.p,
                             act_fn=act_fn,
                             layernorm_eps=layer.LayerNorm.eps,
                             gated_layer_bias=gated_layer_bias,
                             non_gated_layer_bias=non_gated_layer_bias)


def from_BertIntermediate(layer: torch.nn.Module, module_index: int) -> torch.nn.Identity:
    """
    Defines a replacement policy from a :class:`transformers.models.bert.modeling_bert.BertIntermediate` to a :class:`torch.nn.Identity`
    The identity effectively acts as no-op.
    """
    return torch.nn.Identity()


def apply_gated_linear_units(model: torch.nn.Module,
                             optimizers: Union[torch.optim.Optimizer, Sequence[torch.optim.Optimizer]],
                             act_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
                             gated_layer_bias: bool = False,
                             non_gated_layer_bias: bool = False) -> None:
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

    # ensure that the model is an instance of a BERT model, since our replacement policy is only defined for BERTs
    if not isinstance(model, HuggingFaceModel) and not (isinstance(model.model, BertForMaskedLM) or
                                                        isinstance(model.model, BertForSequenceClassification)):
        raise TypeError('Gated Linear Units only has a surgery policy defined for instances of BERT models.')

    if act_fn is None:
        # get the activation functions used
        act_fns = {module.intermediate_act_fn for module in model.modules() if isinstance(module, BertIntermediate)}

        if len(act_fns) != 1:
            raise ValueError('The model has non-uniform activation functions, which is currently unsupported.')

        # since our set is of length-1, let's extract the only activation function remaining.
        (act_fn,) = act_fns

    if act_fn is None:
        raise ValueError(
            'Could not find an existing activation function to use, and no custom activation function was provided.')

    # now that we know the act fn, bind a few parameters of the replacement function
    def from_bound_BertOutput(layer: torch.nn.Module, module_index: int) -> BERTGatedFFOutput:
        return from_BertOutput(layer=layer,
                               module_index=module_index,
                               act_fn=act_fn,
                               gated_layer_bias=gated_layer_bias,
                               non_gated_layer_bias=non_gated_layer_bias)

    # prepare the replacement policy and perform replacement
    policy: Dict[Type[torch.nn.Module], module_surgery.ReplacementFunction] = {
        BertIntermediate: from_BertIntermediate,
        BertOutput: from_bound_BertOutput
    }
    replaced_instances = module_surgery.replace_module_classes(module=model, optimizers=optimizers, policies=policy)
    if len(replaced_instances) == 0:
        warnings.warn(
            NoEffectWarning(
                'No instances of `torch.nn.LayerNorm` were found, and therefore, there were no modules to replace.'))
    log.info(
        f'Successfully replaced {len(replaced_instances)} of BertIntermediate and BertOutput with a GatedLinearUnit.')


class GatedLinearUnits(Algorithm):
    """Replaces all instances of Linear layers in the feed-forward subnetwork with a `Gated Linear Unit <https://arxiv.org/abs/2002.05202>`_.
    The Gated Linear Units provide a more expressive form for the same number of parameters, and a slight degredation to throughput.

    Runs on :attr:`.Event.INIT`, so it can swap the Linear layers in the FFN for GLUs before the model is DDP wrapped.

    Args:
        act_fn (Callable[[torch.Tensor], torch.Tensor], optional): Optionally, the activation function to use. If ``None``, the algorithm will
            use the existing activation function in the model.
        gated_layer_bias (bool, optional): Whether to use biases in the linear layers within the GLU. Default: ``False``.
        non_gated_layer_bias (bool, optional): Whether to use biases in the linear layers within the GLU. Default: ``False``.

    Example:
        .. testsetup::

           model, train_dataloader, optimizer = _make_synthetic_bert_state()

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

    def __init__(self,
                 act_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
                 gated_layer_bias: bool = False,
                 non_gated_layer_bias: bool = False):
        if not IS_TRANSFORMERS_INSTALLED:
            raise MissingConditionalImportError(extra_deps_group='nlp', conda_package='transformers')
        self.act_fn = act_fn
        self.gated_layer_bias = gated_layer_bias
        self.non_gated_layer_bias = non_gated_layer_bias

    def match(self, event: Event, state: State) -> bool:
        del state  # unused
        return event == Event.INIT

    def apply(self, event: Event, state: State, logger: Logger) -> Optional[int]:
        del event, logger  # unused
        apply_gated_linear_units(model=state.model,
                                 optimizers=state.optimizers,
                                 act_fn=self.act_fn,
                                 gated_layer_bias=self.gated_layer_bias,
                                 non_gated_layer_bias=self.non_gated_layer_bias)
