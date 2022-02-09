from composer import Trainer, algorithms, trainer
from composer.core.types import Precision

hparams = trainer.load("gpt2_355m")  # loads from composer/yamls/models/classify_mnist_cpu.yaml

# Create trainer instance
trainer = Trainer.create_from_hparams(hparams)
