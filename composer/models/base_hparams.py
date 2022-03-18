# Copyright 2021 MosaicML. All Rights Reserved.

"""General `YAHP <https://docs.mosaicml.com/projects/yahp/en/stable/README.html>`_ interface for Base ComposerModels."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import torch
import torch.nn as nn
import yahp as hp

from composer.core.types import JSON
from composer.models.base import ComposerModel
from composer.utils.string_enum import StringEnum

__all__ = ["ModelHparams", "Initializer", "TransformerHparams"]


class Initializer(StringEnum):
    """Sets the initialization scheme for different layers of a PyTorch model."""
    KAIMING_NORMAL = "kaiming_normal"
    KAIMING_UNIFORM = "kaiming_uniform"
    BN_UNIFORM = "bn_uniform"
    BN_ONES = "bn_ones"
    XAVIER_UNIFORM = "xavier_uniform"

    def get_initializer(self) -> Callable[[torch.nn.Module], None]:

        def kaiming_normal(w: nn.Module):
            if isinstance(w, torch.nn.Linear) or isinstance(w, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(w.weight)

        def kaiming_uniform(w: nn.Module):
            if isinstance(w, torch.nn.Linear) or isinstance(w, torch.nn.Conv2d):
                torch.nn.init.kaiming_uniform_(w.weight)

        def xavier_uniform(w: nn.Module):
            if isinstance(w, torch.nn.Linear) or isinstance(w, torch.nn.Conv2d):
                torch.nn.init.xavier_uniform_(w.weight)

        def bn_ones(w: nn.Module):
            if isinstance(w, torch.nn.BatchNorm2d):
                w.weight.data = torch.ones_like(w.weight.data)
                w.bias.data = torch.zeros_like(w.bias.data)

        def bn_uniform(w: nn.Module):
            if isinstance(w, torch.nn.BatchNorm2d):
                w.weight.data = torch.rand(w.weight.data.shape)
                w.bias.data = torch.zeros_like(w.bias.data)

        initializer_dict = {
            "kaiming_normal": kaiming_normal,
            "kaiming_uniform": kaiming_uniform,
            "bn_uniform": bn_uniform,
            "bn_ones": bn_ones,
            "xavier_uniform": xavier_uniform
        }
        if self.value not in initializer_dict:
            raise ValueError(f"Initializer '{self.value}' not found.")
        return initializer_dict[self.value]


@dataclass
class ModelHparams(hp.Hparams, ABC):
    """General `YAHP <https://docs.mosaicml.com/projects/yahp/en/stable/README.html>`_ interface for ComposerModels.

    Args:
        num_classes (int): The number of classes. Needed for classification tasks. Default: ``None``.
        initializers (List[Initializer], optional): The initialization strategy for the model. Default: ``None``.
    """
    initializers: List[Initializer] = hp.optional(
        default_factory=lambda: [],
        doc="The initialization strategy for the model",
    )

    num_classes: Optional[int] = hp.optional(
        doc="The number of classes.  Needed for classification tasks",
        default=None,
    )

    @abstractmethod
    def initialize_object(self) -> ComposerModel:
        """Invoked by the :meth:`~composer.trainer.trainer_hparams.TrainerHparams.initialize_object` to construct a
        :class:`.ComposerModel`.

        Returns:
            ComposerModel: The constructed :class:`.ComposerModel`
        """
        pass


@dataclass
class TransformerHparams(ModelHparams, ABC):
    """Defines the necessary hyparameters for a Transformer base module.

    Args:
        pretrained_model_name (str): "Pretrained model name to pull from Huggingface Model Hub."
        model_config (Dict[str, JSON]): A dictionary providing a HuggingFace model configuration.
        tokenizer_name (str): The tokenizer used for this model,
            necessary to assert required model inputs.
        use_pretrained (bool, optional): Whether to initialize the model with the pretrained weights. Default: ``False``
        gradient_checkpointing (bool, optional): Use gradient checkpointing. Default: ``False``.
    """

    tokenizer_name: str = hp.optional("Tokenizer name to pull from Huggingface Model Hub.", default=None)
    pretrained_model_name: Optional[str] = hp.optional(
        doc="Pretrained model name to pull from Huggingface Model Hub.",
        default=None,
    )
    model_config: Dict[str, JSON] = hp.optional(doc="A dictionary providing a HuggingFace model configuration.",
                                                default_factory=dict)
    use_pretrained: bool = hp.optional("Whether to initialize the model with the pretrained weights.", default=False)
    gradient_checkpointing: bool = hp.optional("Whether to enable gradient checkpointing.", default=False)

    def validate(self):
        if self.tokenizer_name is None:
            raise ValueError('tokenizer_name cannot be None. Enter model name to pull from HuggingFace Model Hub.')

        if self.pretrained_model_name is None and self.model_config == {}:
            raise Exception("One of pretrained_model_name or model_config needed.")

        if self.pretrained_model_name is not None and self.model_config != {}:
            raise Exception("Only one of pretrained_model_name or model_config can be provided.")

        if self.use_pretrained and self.model_config:
            raise Exception("A model cannot load pretrained weights from configuration.")