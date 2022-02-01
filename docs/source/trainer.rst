composer.trainer
================

.. currentmodule:: composer

:class:`Trainer` is used to train models with :class:`Algorithm` instances.
The :class:`Trainer` is highly customizable and can support a wide variety of workloads.


Examples
--------

.. code-block:: python

    # Setup dependencies
    from composer.datasets import MNISTDatasetHparams
    from composer.models.mnist import MnistClassifierHparams
    model = MnistClassifierHparams(num_classes=10).initialize_objeect()
    train_dataloader = DataLoader(
        datasets.MNIST('~/datasets/', train=True, transform=transforms.ToTensor(), download=True),
        drop_last=True,
        shuffle=True,
        batch_size=256,
    )

    eval_dataloader = DataLoader(
        datasets.MNIST('~/datasets/', train=True, transform=transforms.ToTensor(), download=True),
        drop_last=False,
        shuffle=False,
        batch_size=256,
    )


.. code-block:: python

    # Create a trainer that will checkpoint every epoch
    # and train the model
    trainer = Trainer(model=model,
                      train_dataloader=train_dataloader,
                      eval_dataloader=eval_dataloader,
                      max_duration="50ep",
                      train_batch_size=128,
                      eval_batch_size=128,
                      checkpoint_interval_unit="ep",
                      checkpoint_folder="checkpoints",
                      checkpoint_interval=1)
    trainer.fit()


.. code-block:: python

    # Load a trainer from the saved checkpoint and resume training
    trainer = Trainer(model=model,
                      train_dataloader=train_dataloader,
                      eval_dataloader=eval_dataloader,
                      max_epochs=50,
                      train_batch_size=128,
                      eval_batch_size=128,
                      checkpoint_filepath="checkpoints/first_checkpoint.pt")
    trainer.fit()


.. code-block:: python

    from composer.trainer import TrainerHparamms

    # Create a trainer from hparams and train train the model
    trainer = Trainer.create_from_hparams(hparams=hparams)
    trainer.fit()


Trainer Hparams
---------------

:class:`Trainer` can be constructed via either it's ``__init__`` (see below)
or
`TrainerHparams <https://github.com/mosaicml/composer/blob/main/composer/trainer/trainer_hparams.py>`_.

Our `yahp <https://github.com/mosaicml/yahp>`_ based system allows configuring the trainer and algorithms via either a ``yaml`` file (see `here <https://github.com/mosaicml/composer/blob/main/composer/yamls/models/classify_mnist_cpu.yaml>`_ for an example) or command-line arguments. Below is a table of all the keys that can be used.

For example, the yaml for ``algorithms`` can include:

.. code-block:: yaml

    algorithms:
        - blurpool
        - layer_freezing


You can also provide overrides at command line:


.. code-block:: bash

    python examples/run_composer_trainer.py -f composer/yamls/models/classify_mnist_cpu.yaml --algorithms blurpool layer_freezing --datadir ~/datasets


**Algorithms**

.. csv-table::
    :header: "name", "algorithm"
    :widths: 20, 40
    :delim: |

    alibi | `AlibiHparams <https://github.com/mosaicml/composer/blob/main/composer/algorithms/alibi/alibi.py>`_
    augmix | `AugMixHparams <https://github.com/mosaicml/composer/blob/main/composer/algorithms/augmix/augmix.py>`_
    blurpool | `BlurPoolHparams <https://github.com/mosaicml/composer/blob/main/composer/algorithms/blurpool/blurpool.py>`_
    channels_last | `ChannelsLastHparams <https://github.com/mosaicml/composer/blob/main/composer/algorithms/channels_last/channels_last.py>`_
    colout | `ColOutHparams <https://github.com/mosaicml/composer/blob/main/composer/algorithms/colout/colout.py>`_
    cutout | `CutOutHparams <https://github.com/mosaicml/composer/blob/main/composer/algorithms/cutout/cutout.py>`_
    dummy | `DummyHparams <https://github.com/mosaicml/composer/blob/main/composer/algorithms/dummy.py>`_
    ghost_batchnorm | `GhostBatchNormHparams <https://github.com/mosaicml/composer/blob/main/composer/algorithms/ghost_batchnorm/ghost_batchnorm.py>`_
    label_smoothing | `LabelSmoothingHparams <https://github.com/mosaicml/composer/blob/main/composer/algorithms/label_smoothing/label_smoothing.py>`_
    layer_freezing | `LayerFreezingHparams <https://github.com/mosaicml/composer/blob/main/composer/algorithms/layer_freezing/layer_freezing.py>`_
    mixup | `MixUpHparams <https://github.com/mosaicml/composer/blob/main/composer/algorithms/mixup/mixup.py>`_
    no_op_model | `NoOpModelHparams <https://github.com/mosaicml/composer/blob/main/composer/algorithms/no_op_model/no_op_model.py>`_
    progressive_resizing | `ProgressiveResizingHparams <https://github.com/mosaicml/composer/blob/main/composer/algorithms/progressive_resizing/progressive_resizing.py>`_
    randaugment | `RandAugmentHparams <https://github.com/mosaicml/composer/blob/main/composer/algorithms/randaugment/randaugment.py>`_
    sam | `SAMHparams <https://github.com/mosaicml/composer/blob/main/composer/algorithms/sam/sam.py>`_
    scale_schedule | `ScaleScheduleHparams <https://github.com/mosaicml/composer/blob/main/composer/algorithms/scale_schedule/scale_schedule.py>`_
    selective_backprop | `SelectiveBackpropHparams <https://github.com/mosaicml/composer/blob/main/composer/algorithms/selective_backprop/selective_backprop.py>`_
    seq_length_warmup | `class SeqLengthWarmupHparams(AlgorithmHparams): <https://github.com/mosaicml/composer/blob/main/composer/algorithms/seq_length_warmup/seq_length_warmup.py>`_
    squeeze_excite | `SqueezeExciteHparams <https://github.com/mosaicml/composer/blob/main/composer/algorithms/squeeze_excite/squeeze_excite.py>`_
    stochastic_depth | `StochasticDepthHparams <https://github.com/mosaicml/composer/blob/main/composer/algorithms/stochastic_depth/stochastic_depth.py>`_
    swa | `SWAHparams <https://github.com/mosaicml/composer/blob/main/composer/algorithms/swa/swa.py>`_

**Callbacks**

.. csv-table::
    :header: "name", "callback"
    :widths: 20, 40
    :delim: |

    benchmarker | :class:`~composer.callbacks.callback_hparams.BenchmarkerHparams`
    grad_monitor | :class:`~composer.callbacks.callback_hparams.GradMonitorHparams`
    lr_monitor | :class:`~composer.callbacks.callback_hparams.LRMonitorHparams`
    speed_monitor | :class:`~composer.callbacks.callback_hparams.SpeedMonitorHparams`

**Datasets**

.. csv-table::
    :header: "name", "dataset"
    :widths: 20, 40
    :delim: |

    brats | :class:`~composer.datasets.BratsDatasetHparams`
    cifar10 | :class:`~composer.datasets.CIFAR10DatasetHparams`
    imagenet | :class:`~composer.datasets.ImagenetDatasetHparams`
    lm | :class:`~composer.datasets.LMDatasetHparams`
    mnist | :class:`~composer.datasets.MNISTDatasetHparams`

**Devices**

.. csv-table::
    :header: "name", "device"
    :widths: 20, 40
    :delim: |

    cpu | `CPUDeviceHparams <https://github.com/mosaicml/composer/tree/main/composer/trainer/devices/device_hparams.py>`_
    gpu | `GPUDeviceHparams <https://github.com/mosaicml/composer/tree/main/composer/trainer/devices/device_hparams.py>`_

**Loggers**

.. csv-table::
    :header: "name", "logger"
    :widths: 20, 40
    :delim: |

    file | :class:`~composer.loggers.FileLoggerBackendHparams`
    tqdm | :class:`~composer.loggers.TQDMLoggerBackendHparams`
    wandb | :class:`~composer.loggers.WandBLoggerBackendHparams`

**Models**

.. csv-table::
    :header: "name", "model"
    :widths: 20, 40
    :delim: |

    efficientnetb0 | `EfficientNetB0Hparams <https://github.com/mosaicml/composer/tree/main/composer/models/efficientnetb0/efficientnetb0_hparams.py>`_
    gpt2 | `GPT2Hparams <https://github.com/mosaicml/composer/blob/main/composer/models/gpt2/gpt2_hparams.py>`_
    mnist_classifier | `MnistClassifierHparams <https://github.com/mosaicml/composer/blob/main/composer/models/classify_mnist/mnist_hparams.py>`_
    resnet18 | `ResNet18Hparams <https://github.com/mosaicml/composer/tree/main/composer/models/resnet18/resnet18_hparams.py>`_
    resnet56_cifar10 | `CIFARResNetHparams <https://github.com/mosaicml/composer/tree/main/composer/models/resnet56_cifar10/resnet56_cifar10_hparams.py>`_
    resnet50 | `ResNet50Hparams <https://github.com/mosaicml/composer/tree/main/composer/models/resnet50/resnet50_hparams.py>`_
    resnet101 | `ResNet101Hparams <https://github.com/mosaicml/composer/tree/main/composer/models/resnet101/resnet101_hparams.py>`_
    unet | `UnetHparams <https://github.com/mosaicml/composer/tree/main/composer/models/unet/unet_hparams.py>`_

**Optimizers**

.. csv-table::
    :header: "name", "optimizer"
    :widths: 20, 40
    :delim: |

    adamw | `AdamWHparams <https://github.com/mosaicml/composer/blob/main/composer/optim/optimizer_hparams.py>`_
    decoupled_adamw | `DecoupledAdamWHparams <https://github.com/mosaicml/composer/blob/main/composer/optim/optimizer_hparams.py>`_
    decoupled_sgdw | `DecoupledSGDWHparams <https://github.com/mosaicml/composer/blob/main/composer/optim/optimizer_hparams.py>`_
    radam | `RAdamHparams <https://github.com/mosaicml/composer/blob/main/composer/optim/optimizer_hparams.py>`_
    rmsprop | `RMSPropHparams <https://github.com/mosaicml/composer/blob/main/composer/optim/optimizer_hparams.py>`_
    sgd | `SGDHparams <https://github.com/mosaicml/composer/blob/main/composer/optim/optimizer_hparams.py>`_

**Schedulers**

.. csv-table::
    :header: "name", "scheduler"
    :widths: 20, 40
    :delim: |

    constant | `ConstantLRHparams <https://github.com/mosaicml/composer/blob/main/composer/optim/scheduler.py>`_
    cosine_decay | `CosineAnnealingLRHparams <https://github.com/mosaicml/composer/blob/main/composer/optim/scheduler.py>`_
    cosine_warmrestart | `CosineAnnealingWarmRestartsHparams <https://github.com/mosaicml/composer/blob/main/composer/optim/scheduler.py>`_
    exponential | `ExponentialLRHparams <https://github.com/mosaicml/composer/blob/main/composer/optim/scheduler.py>`_
    multistep | `MultiStepLRHparams <https://github.com/mosaicml/composer/blob/main/composer/optim/scheduler.py>`_
    step | `StepLRHparams <https://github.com/mosaicml/composer/blob/main/composer/optim/scheduler.py>`_
    warmup | `WarmUpLRHparams <https://github.com/mosaicml/composer/blob/main/composer/optim/scheduler.py>`_


API Reference
-------------

.. autoclass:: Trainer
    :members: