from __future__ import annotations

from typing import TYPE_CHECKING, Mapping

from composer.models.transformer_shared import MosaicTransformer

if TYPE_CHECKING:
    from composer.core.types import Batch, Tensors


class BERTModel(MosaicTransformer):
    """
    Implements a BERT wrapper around a MosaicTransformer.
    """

    def loss(self, outputs: Mapping, batch: Batch) -> Tensors:
        if outputs.get('loss', None) is not None:
            return outputs['loss']
        else:
            raise NotImplementedError('Calculating loss directly not supported yet.')
        

    # TODO (Moin): implement metrics for the validation set
