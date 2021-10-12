from composer.models.base import MosaicClassifier
from composer.models.resnet18.resnet18_hparams import ResNet18Hparams
from composer.models.resnets import ImageNet_ResNet


class Resnet18(MosaicClassifier):

    def __init__(self, hparams: ResNet18Hparams) -> None:
        assert hparams.num_classes is not None, "num_classes must be set in ModelHparams"
        model = ImageNet_ResNet.get_model_from_name(
            "imagenet_resnet_18",
            hparams.initializers,
            hparams.num_classes,
        )
        super().__init__(module=model)
