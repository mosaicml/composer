|:play_or_pause_button:| Auto Resumption
========================================

Resuming from checkpoints is commonly used to recover from hardware failures (e.g. spot instances in the cloud being terminated), loss spikes for large language models, or other unforseen errors. Our trainer supports resuming from checkpoints with the ``load_path`` argument (see :doc:`/trainer/checkpointing` for more details):

.. testsetup::

    import os
    import shutil

    from composer import Trainer

    trainer = Trainer(
        model=model,
        train_dataloader=train_dataloader,
        max_duration="1ep",
        save_filename="ep{epoch}.pt",
        save_folder="./path/to/checkpoints",
        save_overwrite=True,
        save_interval="1ep",  # Save checkpoints every epoch
    )
    trainer.fit()

    assert os.path.exists("./path/to/checkpoints/ep1.pt")

    if not os.path.exists("./path/to/checkpoints/ep25.pt"):
        shutil.copy2("./path/to/checkpoints/ep1.pt", "./path/to/checkpoints/ep25.pt")

    assert os.path.exists("./path/to/checkpoints/ep25.pt")


.. testcode::

    trainer = Trainer(
        model=model,
        train_dataloader=train_dataloader,
        max_duration="90ep",
        save_overwrite=True,
        load_path="./path/to/checkpoints/ep25.pt",
    )
    trainer.fit()

The above code will load the checkpoint from epoch 25 and continue training
for another 65 epochs (to reach 90 epochs total).

However, recovering from a failure here would still require manual intervention to relaunch a new job with the ``load_path`` pointing to the correct checkpoint.

Instead, our trainer supports the ``autoresume=True`` feature. With autoresume, the trainer will automatically check the ``save_folder`` for the latest checkpoints and resume training.

.. testcode::

    trainer = Trainer(
        ...,
        autoresume=True,
        save_folder='./path/to/folder',
        run_name='my_cool_run',
    )

With autoresume, users can re-submit the _same_ code to the training run, and the trainer will handle finding and resuming from the latest checkpoints. This works well with systems like Kubernetes that automatically resubmit the same job when there is a node failure (due to spot instances as well). For ``autoresume=True`` to work, we require that both a ``save_folder`` and a ``run_name`` be provided. These are used to search for existing checkpoints.

For an example code, see the `Checkpoint Autoresumption <examples/checkpoint_autoresume>`_ tutorial.

Implementation
--------------

During training, the trainer always symlinks the latest checkpoint to a format (default is ``latest-rank{rank}`` for local files and ``{run_name}/checkpoints/latest-rank{rank}`` for object stores). When ``autoresume=True``, the Trainer searches for checkpoints of that format in the following order:

1. Local checkpoints of the format ``"{save_folder}/latest-rank0"``. The format for the latest checkpoint can be configured with ``save_latest_filename`` argument (default: ``latest-rank{rank}``).
2. If no local checkpoints are found, then each logger is checked for files of the format ``"{run_name}/checkpoints/latest-rank{rank}"``. This is often used for resuming from an object store such as S3.
3. Finally, ``load_path`` is used to load a checkpoint. This can be used for example, a fine-tuning run on a spot instance, where ``load_path`` would be set to the original weights.

Below, are some examples that demonstrate the object store logger (#2 above) and using the ``load_path`` for fine-tuning purposes (#3 above).

Example: Object Store
---------------------

A typical use case is saving checkpoints to object store (e.g. S3) when there is no local file storage shared across runs. For example, a setup such as this:


.. testcode::

    from composer.loggers import ObjectStoreLogger
    from composer.utils.object_store import S3ObjectStore

    # this assumes credentials are already configured via boto3
    object_store_logger = ObjectStoreLogger(
        object_store_cls=S3ObjectStore,
        object_store_kwargs={
            "bucket": "checkpoint-debugging",
            },
    )

    trainer = Trainer(
        ...,
        autoresume=True,
        save_folder='checkpoints',
        save_num_checkpoints_to_keep=0,  # delete all checkpoints locally
        run_name='my_cool_run',
        save_artifact_name='checkpoints/ep{epoch}.pt',
        loggers=[object_store_logger],
    )

    trainer.fit()


During resumption, there would be no local checkpoints, so the trainer would then look in the object store logger's provided bucket and artifact folder (`checkpoint-debugging/my_cool_run/checkpoints`) to find the latest checkpoint.

Example: Fine-tuning
--------------------

To run fine-tuning on a spot instance, ``load_path`` would be set to the original weights and an object store logger would be added.

.. testsetup:: fine_tune

    from composer.loggers import ObjectStoreLogger
    from composer.utils.object_store import S3ObjectStore

    object_store_logger = ObjectStoreLogger(
        object_store_cls=S3ObjectStore,
        object_store_kwargs={
            "bucket": "checkpoint-debugging_2",
        },
    )

    # Train to generate and save the "pretrained_weights/model.pt",
    # so we can load and resume from it
    trainer = Trainer(
        ...,
        save_filename='pretrained_weights/model.pt',
        save_folder='.',
        run_name='my_cool_run',
    )

    trainer.fit()

.. testcode:: fine_tune

    trainer = Trainer(
        ...,
        autoresume=True,
        load_path='pretrained_weights/model.pt',
        load_weights_only=True,
        save_folder='checkpoints',
        run_name='my_cool_run',
        loggers=[
            object_store_logger
        ]
    )


In the original run, ``load_path`` would be used to get the starting checkpoint. For any future restarts, such as due to the spot instance being terminated, the loggers would be queried for the latest checkpoint the object store logger would be downloaded and used to resume training, and the ``load_path`` would be ignored.

.. note::

    The pretrained weights can also be loaded from object store with the trainer's ``load_object_store`` argument. In that way, our trainer is fully independent of any local storage!
