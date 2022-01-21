# Copyright 2021 MosaicML. All Rights Reserved.
from typing import Optional

from dataclasses import dataclass
import yahp as hp
from composer.models.model_hparams import ModelHparams
import timm


@dataclass
class TimmHparams(ModelHparams):

    model_name: str = hp.optional(
        "timm model name e.g:  list of models can be found at https://github.com/rwightman/pytorch-image-models",
        default=None,
    )
    pretrained: bool = hp.optional("imagenet pretrained", default=False)
    num_classes: int = hp.optional(
        "The number of classes.  Needed for classification tasks", default=1000
    )
    drop_rate: float = hp.optional("dropout rate", default=0.0)
    drop_path_rate: Optional[float] = hp.optional(
        "drop path rate (model default if None)", default=None
    )
    drop_block_rate: Optional[float] = hp.optional(
        "drop block rate (model default if None)", default=None
    )
    global_pool: Optional[str] = hp.optional(
        "Global pool type, one of (fast, avg, max, avgmax, avgmaxc). Model default if None.",
        default=None
    )
    bn_momentum: Optional[float] = hp.optional(
        "BatchNorm momentum override (model default if not None)", default=None
    )
    bn_eps: Optional[float] = hp.optional(
        "BatchNorm epsilon override (model default if not None)", default=None
    )

    def validate(self):
        if self.model is None:
            raise ValueError(f"model must be one of {timm.models.list_models()}")

    def initialize_object(self):
        return timm.create_model(
            model_name=self.model_name,
            pretrained=self.pretrained,
            num_classes=self.num_classes,
            drop_rate=self.drop_rate,
            drop_path_rate=self.drop_path_rate,
            drop_block_rate=self.drop_block_rate,
            global_pool=self.global_pool,
            bn_momentum=self.bn_momentum,
            bn_eps=self.bn_eps)
