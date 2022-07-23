# 🏃‍♀️ Run Name

The `run_name` is a string used to name a specific training run. Naming your training run has many benefits.
Namely, you can more easily group and keep track of metrics, checkpoints, and other training artifacts.
In addition, your `run_name` will show up in many places as you utilize Composer.

## Run Name Creation

The `run_name` argument is an optional argument to the  {class}`~.Trainer`.

There are two ways to get a `run_name`.

You create your own `run_name` and pass it to the trainer, like so:

<!--pytest-codeblocks:skip-->
```python
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from composer import Trainer
from composer.models import mnist_model

transform = transforms.Compose([transforms.ToTensor()])
dataset = datasets.MNIST("data", train=True, download=True, transform=transform)
train_dataloader = DataLoader(dataset, batch_size=128)
run_name = 'my-cool-run-name'

trainer = Trainer(
    model=mnist_model(num_classes=10),
    train_dataloader=train_dataloader,
    max_duration="2ep",
    run_name=run_name,
)
trainer.fit()
```

You can instead let the trainer create a `run_name` for you. The one created for you is the timestamp followed by a [coolname](https://github.com/alexanderlukanin13/coolname); e.g. `‘1657932618-infrared-ferret’`


## How the Run Name is Used

This run_name will be added as an attribute to {class}`~.State` and it is used by various other pieces of the composer infrastructure as described below.

The `run_name` is often used in the Composer as placeholder in a format string. This means that if a string is specified to name a file, like `'{run_name}-foo-bar'` for example, then that placeholder will get filled in by the actual `run_name` at runtime, so the file will actually be named `‘1657932618-infrared-ferret-foo-bar’`.

### Run Names in Checkpoint Saving

In checkpoint saving you can use the `run_name` as a placeholder in a format string to name the folders and checkpoints locally and in the cloud if you are uploading your checkpoints using Weights and Biases or an ObjectStoreLogger.

See {class}`~.CheckpointSaver` for more information on specifying the arguments for files and folder names with the `run_name` when creating a {class}`~.Trainer` object.

### Run Names in Logging

In addition to checkpointing, loggers also use the `run_name` for default logging.

#### Tensorboard Logger

The {class}`~.TensorboardLogger` will save all the logs for a run to a folder called `run_name` and the name of each run in the Tensorboard GUI will be `run_name`.

#### Weights and Biases Logger

The `run_name` you specify will be used by the {class}`~.WandBLogger` as the run name for Weights and Biases.

#### Object Store Logger

The {class}`~.ObjectStoreLogger` will often use the `run_name` as part of how it names objects.

#### File Logger

The `run_name` is also used in the {class}`~.FileLogger` as the default name for the file the {class}`~.FileLogger` logs to is `'{run_name}/logs-rank{rank}.txt'`


See [Logging](../trainer/logging.rst) for more information.

### Run Names in Profiling

The profiling tools for the training also save profiling artifacts to folders named after the `run_name`.  See [Performance Profiling](../trainer/performance_tutorials/profiling.md) for more information.
