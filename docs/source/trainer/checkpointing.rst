|:white_check_mark:| Checkpointing
==================================

Composer can be configured to automatically save training checkpoints by passing the argument ``save_folder`` when
creating the :class:`.Trainer`.

To customize the filenames of checkpoints inside ``save_folder``, you can set the ``save_filename`` argument.
By default, checkpoints will be named like ``'ep{epoch}-ba{batch}-rank{rank}'`` within the ``save_folder``.

In addition, the trainer creates a symlink called ``'latest-rank{rank}'``, which points to the latest saved checkpoint
file. You can customize this symlink name by setting the ``save_latest_filename`` argument.

The ``save_folder``, ``save_filename``, and ``save_latest`` arguments are Python format strings, so you can customize the folder
structure to include information such as the rank of the Python process or the current training progress. Please see
the :class:`~.CheckpointSaver` for the full list of available format variables.

For example:

.. testcode::

    from composer import Trainer

    trainer = Trainer(
        model=model,
        train_dataloader=train_dataloader,
        max_duration="2ep",
        save_folder="./path/to/checkpoints",
        save_filename="ep{epoch}",
        save_latest_filename="latest",
        save_overwrite=True,
    )

    trainer.fit()

Save Interval
-------------

By default, checkpoints are saved every epoch, but this interval can be configured using the ``save_interval`` argument.
The ``save_interval`` can be an integer (interpreted as a number of epochs), a time string (see the
:doc:`Time Guide </trainer/time>` for more information), or a function that takes
(:class:`~.State`, :class:`~.Event`) and returns whether a checkpoint should be saved.

For example:

*   ``save_interval=1`` to save every epoch (the default).
*   ``save_interval="10ep"`` to save every 10 epochs.
*   ``save_interval="500ba"`` to save every 500 batches/steps.
*   ``save_interval=lambda state, event: state.timestamp.epoch > 50 and event == Event.EPOCH_CHECKPOINT``
    to save every epoch, starting after the 50th epoch.

Putting this together, here's how to save checkpoints:

.. testcode::

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

The above code will train a model for 1 epoch, and then save the checkpoint.

Anatomy of a Checkpoint
-----------------------

The above code, when run, will produce the checkpoints below:

.. doctest::

    >>> trainer.saved_checkpoints
    ['./path/to/checkpoints/ep1.pt']
    >>> latest_checkpoint = trainer.saved_checkpoints[-1]
    >>> state_dict = torch.load(latest_checkpoint)
    >>> list(state_dict)
    ['state', 'rng']
    >>> list(state_dict['state'].keys())
    ['model', 'optimizers', 'schedulers', 'algorithms', 'callbacks', 'scaler', 'timestamp', 'rank_zero_seed', 'train_metrics', 'eval_metrics', 'run_name', 'dataset_state', 'integrations', 'metadata']

Resume training
---------------

To resume training from a previous checkpoint, set the ``load_path`` argument of the :class:`.Trainer` to the checkpoint
filepath.  When the :class:`.Trainer` is initialized, the checkpoint state will be restored, and the :meth:`.Trainer.fit`
will continue training from where the checkpoint left off.

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

Different ``model`` or ``optimizer`` objects that are passed into the trainer when it is
resumed will be respected. However, an error will be raised if the weights or
state from the checkpoint are not compatible with these new objects.


.. note::

    Only the attributes in :attr:`.State.serialized_attributes` are serialized and loaded. By default, they are:

    +-----------------------+-------------------------------------------------------------+
    | Attribute             | Description                                                 |
    +=======================+=============================================================+
    | model                 | The model under training.                                   |
    +-----------------------+-------------------------------------------------------------+
    | optimizers            | The optimizers being used to train the model.               |
    +-----------------------+-------------------------------------------------------------+
    | schedulers            | The learning rate schedulers.                               |
    +-----------------------+-------------------------------------------------------------+
    | algorithms            | The algorithms used for training.                           |
    +-----------------------+-------------------------------------------------------------+
    | callbacks             | The callbacks used for training.                            |
    +-----------------------+-------------------------------------------------------------+
    | scaler                | The gradient scaler in use for mixed precision training.    |
    +-----------------------+-------------------------------------------------------------+
    | timestamp             | The timestamp that tracks training loop progress.           |
    +-----------------------+-------------------------------------------------------------+
    | rank_zero_seed        | The seed of the rank zero process.                          |
    +-----------------------+-------------------------------------------------------------+
    | train_metrics         | The current training metrics.                               |
    +-----------------------+-------------------------------------------------------------+
    | eval_metrics          | The current validation metrics.                             |
    +-----------------------+-------------------------------------------------------------+
    | run_name              | The run name for training.                                  |
    +-----------------------+-------------------------------------------------------------+
    | dataset_state         | The dataset iteration state.                                |
    +-----------------------+-------------------------------------------------------------+

    All other trainer arguments (e.g. ``max_duration`` or ``precision``) will use either the defaults
    or what is passed in when reconstructing the trainer.


Saving for Inference
--------------------

By default, the :class:`.Trainer` stores the entire training state in each checkpoint. If you would like to store
only the model weights in a checkpoint, set ``save_weights_only=True``.

.. testcode::

    from composer.trainer import Trainer

    trainer = Trainer(
        ...,
        save_folder="checkpoints",
        save_weights_only=True,
        save_overwrite=True,
    )

    trainer.fit()

Saving Multiple Checkpoint Types
--------------------------------

To save multiple checkpoint types, such as full checkpoints and weights-only checkpoints, the
:class:`~.CheckpointSaver` can be passed directly into the ``callbacks`` argument of the trainer.
Each :class:`~.CheckpointSaver` can have its own save folder, interval, and other parameters.

When configuring checkpoints via the ``callbacks``, it is not necessary to specify the ``save_folder``
or other checkpoint saving parameters directly on the trainer.

.. testcode::

    from composer.trainer import Trainer
    from composer.callbacks import CheckpointSaver

    trainer = Trainer(
        ...,
        callbacks=[
            CheckpointSaver(
                folder='full_checkpoints',
                save_interval='5ep',
                overwrite=True,
                num_checkpoints_to_keep=1,  # only keep the latest, full checkpoint
            ),
            CheckpointSaver(
                folder='weights_only_checkpoints',
                weights_only=True,
                overwrite=True,
            ),
        ],
    )

    trainer.fit()

Fine-tuning
-----------

The :class:`.Trainer` will only load the model weights from the checkpoint if ``load_weights_only=True`` or if the
checkpoint was saved with ``save_weights_only=True``. This is especially useful for model fine-tuning, since the rest
of the trainer's state no longer applies.

If the fine-tuned model contains different parameter names than the model in the checkpoint, set ``load_strict=False`` to
ignore mismatches in model parameter names between the serialized model state and new model object.
Parameters with the same name are expected to have the same shape and will have their state restored.
Parameters with different names will ignored.

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

    if not os.path.exists("./path/to/checkpoints/ep50.pt"):
        shutil.copy2("./path/to/checkpoints/ep1.pt", "./path/to/checkpoints/ep50.pt")

    assert os.path.exists("./path/to/checkpoints/ep50.pt")

    finetune_model = model
    finetune_dataloader = train_dataloader

.. testcode::

    ft_trainer = Trainer(
        model=finetune_model,
        train_dataloader=finetune_dataloader,
        max_duration="10ep",
        load_path="./path/to/checkpoints/ep50.pt",
        load_weights_only=True,
        load_strict_model_weights=False,
    )

    ft_trainer.fit()

This example will load only the model weights from epoch 1 and then continue training on the fine-tuned dataloader
for 10 epochs.

Loading Weights Externally
--------------------------

The model weights are located at ``state_dict["state"]["model"]`` within the stored checkpoint. To load them into a
model outside of a :class:`.Trainer`, use :meth:`torch.load`:

.. testcode::

    model = Model(num_channels, num_classes)
    state_dict = torch.load("./path/to/checkpoints/ep1.pt")
    model.load_state_dict(state_dict["state"]["model"])

Uploading Checkpoints to Object Store
-------------------------------------

Checkpoints can also be saved to and loaded from your object store of choice (e.g. AWS S3 or Google Cloud Storage).
Writing checkpoints to an object store is a two-step process. The checkpoints are first written to the local filesystem,
and then the :class:`.RemoteUploaderDownloader` logger will upload checkpoints to the specified object store.

Behind the scenes, the :class:`.RemoteUploaderDownloader` uses :doc:`Apache Libcloud <libcloud:storage/index>`.

The easiest way to upload checkpoints to S3, OCI, or GCS is to prefix your ``save_folder``  with ``s3://``, ``oci://``, or ``gs://``. All other
checkpoint arguments remain the same. For example, ``save_filename`` will be the name of the checkpoint file
that gets uploaded to the S3, OCI, or GCS URI that you specified.

For example, for S3:

.. testcode::
    :skipif: not _LIBCLOUD_INSTALLED

    from composer.trainer import Trainer

    # Save checkpoints every epoch to s3://my_bucket/checkpoints
    trainer = Trainer(
        model=model,
        train_dataloader=train_dataloader,
        max_duration='10ep',
        save_folder='s3://my_bucket/checkpoints',
        save_interval='1ep',
        save_overwrite=True,
        save_filename='ep{epoch}.pt',
        save_num_checkpoints_to_keep=0,  # delete all checkpoints locally
    )

    trainer.fit()

This will train your model, saving the checkpoints locally, upload them to the S3 Bucket ``my_bucket``,
and delete the checkpoints from the local disk. The checkpoints will be located on S3 inside your bucket as
``checkpoints/ep3.pt`` for third epoch's checkpoints, for example. The full URI in this case would be:
``s3://my_bucket/checkpoints/ep3.pt``.

For uploading checkpoints to [Coreweave's object store](https://docs.coreweave.com/storage/object-storage), the code is very similar to the
above S3 uploading code. The only difference is you must set your Coreweave endpoint url.
To do this you can just set the ``S3_ENDPOINT_URL`` environment variable before creating the
:class:`.Trainer`, like so:

.. testcode::
    :skipif: not _LIBCLOUD_INSTALLED

    import os

    os.environ['S3_ENDPOINT_URL'] = 'https://object.las1.coreweave.com'
    from composer.trainer import Trainer

    # Save checkpoints every epoch to s3://my_bucket/checkpoints
    trainer = Trainer(
        model=model,
        train_dataloader=train_dataloader,
        max_duration='10ep',
        save_folder='s3://my_bucket/checkpoints',
        save_interval='1ep',
        save_overwrite=True,
        save_filename='ep{epoch}.pt',
        save_num_checkpoints_to_keep=0,  # delete all checkpoints locally
    )

    trainer.fit()


Similarly for OCI:

.. testcode::
    :skipif: not _LIBCLOUD_INSTALLED

    from composer.trainer import Trainer

    # Save checkpoints every epoch to oci://my_bucket/checkpoints
    trainer = Trainer(
        model=model,
        train_dataloader=train_dataloader,
        max_duration='10ep',
        save_folder='oci://my_bucket/checkpoints',
        save_interval='1ep',
        save_overwrite=True,
        save_filename='ep{epoch}.pt',
        save_num_checkpoints_to_keep=0,  # delete all checkpoints locally
    )

    trainer.fit()

This will train your model, saving the checkpoints locally, upload them to the OCI Bucket ``my_bucket``,
and delete the checkpoints from the local disk. The checkpoints will be located on OCI inside your bucket as
``checkpoints/ep3.pt`` for third epoch's checkpoints, for example. The full URI in this case would be:
``oci://my_bucket/checkpoints/ep3.pt``.

Similarly for GCS:

.. testcode::
    :skipif: not _LIBCLOUD_INSTALLED

    from composer.trainer import Trainer

    # Save checkpoints every epoch to gs://my_bucket/checkpoints
    trainer = Trainer(
        model=model,
        train_dataloader=train_dataloader,
        max_duration='10ep',
        save_folder='gs://my_bucket/checkpoints',
        save_interval='1ep',
        save_overwrite=True,
        save_filename='ep{epoch}.pt',
        save_num_checkpoints_to_keep=0,  # delete all checkpoints locally
    )

    trainer.fit()

This will train your model, saving the checkpoints locally, upload them to the GCS Bucket ``my_bucket``,
and delete the checkpoints from the local disk. The checkpoints will be located on GCS inside your bucket as
``checkpoints/ep3.pt`` for third epoch's checkpoints, for example. The full URI in this case would be:
``gs://my_bucket/checkpoints/ep3.pt``.
Note: For GCS, remember to input your `HMAC access id and secret <https://cloud.google.com/storage/docs/authentication/hmackeys/>`__
to the environment variables ``GCS_KEY`` and ``GCS_SECRET`` respectively or the save operation will fail.

There are a few additional trainer arguments which can be helpful to configure:

*   ``save_num_checkpoints_to_keep``: Set this parameter to remove checkpoints from the local disk after they have been
    uploaded. For example, setting this parameter to 1 will only keep the latest checkpoint locally; setting it to 0
    will remove each checkpoint after it has been uploaded. Checkpoints are never deleted from object stores.
*   ``save_remote_file_name``: To customize how checkpoints are named in the cloud bucket, modify this parameter. By
    default, they will be named as ``'{run_name}/checkpoints/ep{epoch}-ba{batch}-rank{rank}'``. See the
    :class:`.CheckpointSaver` documentation for the available format variables.

This is equivalent to creating a RemoteUploaderDownloader object and adding it to loggers. This a more
involved operation, but is necessary for uploading checkpoints to other cloud object stores not supported by URI

.. testcode::
    :skipif: not _LIBCLOUD_INSTALLED

    from composer.loggers import RemoteUploaderDownloader
    from composer.trainer import Trainer

    remote_uploader_downloader = RemoteUploaderDownloader(
        bucket_uri="libcloud://checkpoint-debugging",
        backend_kwargs={
            "provider": "s3",  # The Apache Libcloud provider name
            "container": "checkpoint-debugging",  # The name of the cloud container (i.e. bucket) to use.
            "provider_kwargs": {  # The Apache Libcloud provider driver initialization arguments
                'key': 'provider_key',  # The cloud provider key.
                'secret': '*******',  # The cloud provider secret.
                # Any additional arguments required for the cloud provider.
            },
        },
    )

    trainer = Trainer(
        model=model,
        train_dataloader=train_dataloader,
        max_duration='10ep',
        save_folder='checkpoints',
        save_interval='1ep',
        save_overwrite=True,
        save_filename='ep{epoch}.pt',
        save_num_checkpoints_to_keep=0,  # delete all checkpoints locally
        loggers=[remote_uploader_downloader],
    )

    trainer.fit()

.. seealso::

    *   :doc:`Full list of object store providers <libcloud:storage/supported_providers>`
    *   :class:`~.RemoteUploaderDownloader`


Loading Checkpoints from Object Store
-------------------------------------

Checkpoints saved to an object store can also be loaded in the same way as files saved on disk. Provide the
:class:`.LibcloudObjectStore` to the trainer's ``load_object_store`` argument (you can also provide the full
:class:`.RemoteUploaderDownloader` object as well). The ``load_path`` argument
should be the path to the checkpoint file *within the container/bucket*.

.. testcode::
    :skipif: not _LIBCLOUD_INSTALLED

    from composer.utils import LibcloudObjectStore
    from composer.trainer import Trainer

    object_store = LibcloudObjectStore(
        provider="s3",  # The Apache Libcloud provider name
        container="checkpoint-debugging",  # The name of the cloud container (i.e. bucket) to use.
        provider_kwargs={  # The Apache Libcloud provider driver initialization arguments
            'key': 'provider_key',  # The cloud provider key.
            'secret': '*******',  # The cloud provider secret.
            # Any additional arguments required for the cloud provider.
        },
    )

    new_trainer = Trainer(
        model=model,
        train_dataloader=train_dataloader,
        max_duration="10ep",
        load_path="checkpoints/ep1.pt",
        load_object_store=object_store,
    )

    new_trainer.fit()

An easier way to load checkpoints from S3, OCI, GCS specifically is to just use a URI starting with ``s3://``, ``oci://``, or ``gs://``.
If you use the S3, OCI, or GCS URI, it is not necessary to specify a ``load_object_store``. Note, that for other
object stores like WandB or LibCloud, you must still specify a ``load_object_store``.

.. testcode::
    :skipif: not _LIBCLOUD_INSTALLED

    new_trainer = Trainer(
    model=model,
    train_dataloader=train_dataloader,
    max_duration="10ep",
    load_path="s3://checkpoint-debugging/checkpoints/ep1.pt",
    )

    new_trainer.fit()

This will load the first epoch's checkpoints from S3 and resume training in the second epoch.

Similarly for OCI:

.. testcode::
    :skipif: not _LIBCLOUD_INSTALLED

    new_trainer = Trainer(
    model=model,
    train_dataloader=train_dataloader,
    max_duration="10ep",
    load_path="oci://checkpoint-debugging/checkpoints/ep1.pt",
    )

    new_trainer.fit()

This will load the first epoch's checkpoints from OCI and resume training in the second epoch.


Similarly for GCS:

.. testcode::
    :skipif: not _LIBCLOUD_INSTALLED

    new_trainer = Trainer(
    model=model,
    train_dataloader=train_dataloader,
    max_duration="10ep",
    load_path="gs://checkpoint-debugging/checkpoints/ep1.pt",
    )

    new_trainer.fit()

This will load the first epoch's checkpoints from GCS and resume training in the second epoch.
Note: For GCS, remember to input your `HMAC access id and secret <https://cloud.google.com/storage/docs/authentication/hmackeys/>`__
to the environment variables ``GCS_KEY`` and ``GCS_SECRET`` respectively or the save operation will fail.

.. warning::
    Do not load checkpoints from untrusted sources as they may contain malicious code.
    Users should ensure the proper sanity checks are in place before loading checkpoints.


Saving and Loading Sharded Checkpoints
--------------------------------------
Using `FSDP <https://pytorch.org/docs/stable/fsdp.html>`__, you can save and load sharded checkpoints with Composer.
See `these docs </notes/distributed_training.html#saving-and-loading-sharded-checkpoints-with-fsdp>`_ for more info and some examples.


API Reference
-------------
*   :class:`.RemoteUploaderDownloader` for saving checkpoints to cloud storage.
*   :class:`.Trainer` for the trainer checkpoint arguments.
*   :class:`.CheckpointSaver` for the CheckpointSaver arguments.
*   :class:`.LibcloudObjectStore` for setting up libcloud-supported object stores.
*   :mod:`composer.utils.checkpoint` for the underlying utilities to manually save and load checkpoints.
