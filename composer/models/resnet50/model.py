from composer.models.base import MosaicClassifier
from composer.models.resnet50.resnet50_hparams import ResNet50Hparams
from composer.models.resnets import ImageNet_ResNet


class ResNet50(MosaicClassifier):

    def __init__(self, hparams: ResNet50Hparams) -> None:
        assert hparams.num_classes is not None, "num_classes must be set in ModelHparams"
        model = ImageNet_ResNet.get_model_from_name(
            "imagenet_resnet_50",
            hparams.initializers,
            hparams.num_classes,
        )
        super().__init__(module=model)
