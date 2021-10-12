# MosaicML Composer

MosaicML Composer contains a library of methods, and ways to compose them together for more efficient ML training. We aim to ease the transition from research to industry through reproducible code and rigorous benchmarking.

TODO: add link to our blog post and research philosophy

Use Composer in a few ways:

1. Import the functional form of our methods:

```python
import composer.functional as CF

# replaces eligible layers with BlurPool (cite here)
my_model = CF.apply_blurpool(my_model)

for epoch in range(max_epochs):
    for data in your_data:
        ...
    # freeze layers at the end of every epoch
    CF.freeze_layers(my_model)

```

We have a growing collection of deeply characterized methods, see <todo link>

2. Compose methods together by using the `MosaicTrainer`:

```python
from composer import trainer, algorithms

trainer_hparams = trainer.load("resnet50")
trainer_hparams.algorithms = algorithms.load_multiple(["squeeze_excite", "scale_schedule"])
trainer = trainer.MosaicTrainer.create_from_hparams(hparams=trainer_hparams)
trainer.fit()

```

3. Compose methods together with our `engine` and `events` in your trainer:

```python
from composer import Engine, State, Event, simple_defaults
from composer.algorithms import ChannelsLast, ProgressiveResizing, ScaleSchedule

algorithms = [
    ChannelsLast(),
    ProgressiveResizing(),
    ScaleSchedule()
]

state = State(model=my_model, **simple_defaults)  # stores the state of your trainer
engine = Engine(state=state, algorithms=algorithms)

engine.run_algorithms(Event.TRAINING_START)

for data in your_dataloader:
    # run algorithms at relevant events in your loop
    engine.run_algorithms(Event.AFTER_DATALOADER)
    ...


```

For an example, see byo_trainer.py.

## Composer tl;dr

Composer methods are either curated from the literature, or developed internally, and rigorously measured on public benchmarks. To explore the benchmarks, see our [MosaicML Explorer](https://app.mosaicml.com).

To compose methods together, we used the excellent two-way callbacks system (cite Howard et al) from FastAI. Each method is implemented as a two-way callback, and also in functional form for standalone access and extension.

## Quick start

Install the library with:

```bash
pip install mosaicml-composer
```

TODO in codebase: add import error messages to use various tags

TODO: add content here

## Community

To contribute new methods, models, and datasets, see <TODO: contribution guide link>. Also join our [community slack](https://join.slack.com/t/mosaicmlworkspace/shared_invite/zt-w0tiddn9-WGTlRpfjcO9J5jyrMub1dg) to talk about ML training efficiency!


## Acknowledgements

Our library builds upon ideas from the broader ML community! Composing methods together is rooted in the two-way callbacks from FastAI; model surgery is extended from the PyTorch MKLDNN library; neatly restoring dataloader state from HuggingFace, and we use `torchmetrics` from PyTorch Lightning. We are exploring integrations into all of these libraries to make the Composer efficiency methods available to all.
