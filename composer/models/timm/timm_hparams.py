# Copyright 2021 MosaicML. All Rights Reserved.
import textwrap
from dataclasses import dataclass
from typing import Optional

import yahp as hp

from composer.models.model_hparams import ModelHparams
from composer.models.timm.model import Timm


@dataclass
class TimmHparams(ModelHparams):

    model_name: str = hp.optional(
        textwrap.dedent("""\
        timm model name e.g: 'resnet50', list of models can be found at
        https://github.com/rwightman/pytorch-image-models"""),
        default=None,
    )
    pretrained: bool = hp.optional("imagenet pretrained", default=False)
    num_classes: int = hp.optional("The number of classes.  Needed for classification tasks", default=1000)
    drop_rate: float = hp.optional("dropout rate", default=0.0)
    drop_path_rate: Optional[float] = hp.optional("drop path rate (model default if None)", default=None)
    drop_block_rate: Optional[float] = hp.optional("drop block rate (model default if None)", default=None)
    global_pool: Optional[str] = hp.optional(
        "Global pool type, one of (fast, avg, max, avgmax, avgmaxc). Model default if None.", default=None)
    bn_momentum: Optional[float] = hp.optional("BatchNorm momentum override (model default if not None)", default=None)
    bn_eps: Optional[float] = hp.optional("BatchNorm epsilon override (model default if not None)", default=None)

    def validate(self):
        if self.model_name is None:
            try:
                import timm
            except ImportError as e:
                raise ImportError(
                    textwrap.dedent("""\
                    Composer was installed without timm support. To use timm with Composer, run `pip install mosaicml[timm]`
                    if using pip or `pip install timm>=0.5.4` if using Anaconda.""")) from e
            raise ValueError(f"model must be one of {timm.models.list_models()}")

    def initialize_object(self):
        return Timm(model_name=self.model_name,
                    pretrained=self.pretrained,
                    num_classes=self.num_classes,
                    drop_rate=self.drop_rate,
                    drop_path_rate=self.drop_path_rate,
                    drop_block_rate=self.drop_block_rate,
                    global_pool=self.global_pool,
                    bn_momentum=self.bn_momentum,
                    bn_eps=self.bn_eps)
