# MosaicML Composer

MosaicML `Composer` contains a library of methods, and ways to compose them together for more efficient ML training. We aim to ease the transition from research to industry through reproducible code and rigorous benchmarking.

The library features:
* Implementation of 20+ efficiency methods curated from the research community
* Standardized approach to implement and compose efficiency methods, extended from two-way callbacks ([Howard et al, 2020](https://arxiv.org/abs/2002.04688))
* Easy way to access our methods either directly for your trainer loops, or through the MosaicML Trainer.

To install `Composer`:
```
pip install mosaicml
```

A few ways to use `Composer`:

1. Import the functional form of our methods:

```python
from composer import functional as CF
import torchvision

model = torchvision.models.resnet50()

# replaces eligible layers with BlurPool (cite here)
CF.apply_blurpool(model)

for epoch in range(max_epochs):
    for data in your_data:
        ...
    # freeze layers at the end of every epoch
    CF.freeze_layers(model)

```

We have a growing collection of deeply characterized methods, see [Methods](https://www.mosaicml.com/methods).

2. Compose methods together using our `Trainer`:

```python
from composer import trainer, algorithms, Trainer

trainer_hparams = trainer.load("resnet50")
trainer_hparams.algorithms = algorithms.load_multiple("squeeze_excite", "scale_schedule")
trainer_hparams.set_datadir('your/dataset/path/')

learner = Trainer.create_from_hparams(hparams=trainer_hparams)
learner.fit()

```

## Composer TL;DR

Composer methods are either curated from the literature, or developed internally, and rigorously measured on public benchmarks. To explore the benchmarks, see our [MosaicML Explorer](https://app.mosaicml.com).

To compose methods together, we used the excellent two-way callbacks system ([Howard et al, 2020](https://arxiv.org/abs/2002.04688)). Each method is implemented as a two-way callback, and also in functional form for standalone access and extension.

## Documentation

See [our documentation](https://mosaicml-composer.readthedocs-hosted.com/en/stable/) for installation instructions and how to get started.

## Community

We welcome contributions of new methods, models, and datasets Also join our [community slack](https://join.slack.com/t/mosaicmlworkspace/shared_invite/zt-w0tiddn9-WGTlRpfjcO9J5jyrMub1dg) to talk about ML training efficiency!


Our library builds upon ideas from the broader ML community! We are exploring integrations into other libraries to make the Composer efficiency methods available to all.
