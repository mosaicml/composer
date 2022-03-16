|:white_check_mark:| Checkpointing
==================================

Composer can be configured to automatically save training checkpoints by
passing the argument ``save_folder`` when creating the
:class:`.Trainer`. The ``save_folder`` can be a relative path, in which case
checkpoints will be stored in
``CWD/runs/<timestamp>/<rank>/<save_folder>``. Absolute paths will be used as-is.

By default, checkpoints are saved every epoch, but can be configured
using the ``save_interval`` argument. Specify ``save_interval="10ep"``
to save every 10 epochs or ``save_interval="500ba"`` to save every
500 batches/steps.

.. code:: python

   from composer import Trainer

   trainer = Trainer(model=model,
                     train_dataloader=dataloader,
                     max_duration="1ep",
                     save_folder="/path/to/checkpoints",
                     save_interval="1ep")  # Save checkpoints every epoch
   trainer.fit()

The above code will train a model for 1 epoch, and then save the checkpoint.

Anatomy of a checkpoint
-----------------------

The above code, when run, will produce the checkpoints below:

.. code:: python

   os.listdir(trainer.checkpoint_saver.checkpoint_folder)

   ['ep1.pt']

Opening one of those checkpoints, you'll see:

.. code:: python

   state_dict = torch.load(
       os.path.join(trainer.checkpoint_saver.checkpoint_folder, "ep1.pt")
   )
   print(f"Top level keys: {list(state_dict.keys())}")
   print(f"state keys: {list(state_dict['state'].keys())}")

   >>> Top level keys: ['rng', 'state']
   >>> Keys: ['model', 'timer', 'optimizers', 'schedulers', 'scaler', 'algorithms', 'callbacks', 'rng', 'rank_zero_seed', 'is_model_ddp']

At the top level, we see details on the current RNG state and the
``trainer.state``.

Under the ``"state"`` key, we see:

1. ``"model"``: Model weights
2. ``"_optimizers"``: Optimizer state
3. ``"_schedulers"``: Scheduler state
4. ``"_algorithms"``: Any algorithm state

These are the most important keys to be aware of. There are several
others that are required to ensure that you can pick back up where you
left off.

Resume training
---------------

To resume training from a previous checkpoint, pass the
checkpoint file path to the :class:`.Trainer` with the
``load_path_format`` argument. This should be an absolute path.

When the :class:`.Trainer` is initialized, all the state
information will be restored from the checkpoint and
``trainer.fit()`` will continue training from where the checkpoint left off.

.. code:: python

   trainer = Trainer(model=model,
                     train_dataloader=dataloader,
                     eval_dataloader=None,
                     max_duration="90ep",
                     load_path_format="/path/to/checkpoint/ep25.pt")
   trainer.fit()

The above code will load the checkpoint from epoch 25, and continue training
for another 65 epochs (to reach 90 epochs total).

Different ``model`` or ``optimizer`` objects passed into the trainer when
resume will be respected. However, an error will be raised if the weights or
state from the checkpoint are not compatible with these new objects.

..note ::

    Only the following attributes from :class:`.State` will be serialized and loaded:

    .. code:: python

        serialized_attributes = [
                "model",
                "optimizers",
                "schedulers",
                "algorithms",
                "callbacks",
                "scaler",
                "timer",
            ]

    All other trainer arguments (e.g. ``max_duration`` or ``precision``) will use
    the defaults or what is passed in during the trainer creation.



Fine-tuning
-----------

The :class:`.Trainer` will only load the model weights from the checkpoint if
``load_weights_only=True``. This is especially useful for model finetuning,
since the rest of the trainer's state no longer applies.

.. code:: python

   ft_trainer = Trainer(model=model,
                        train_dataloader=finetune_dataloader,
                        eval_dataloader=None,
                        max_duration="10ep",
                        load_path_format="/path/to/checkpoint/ep50.pt",
                        load_weights_only=True)

This example will load only the model weights from epoch 50, and then continue
training on the finetuned dataloader for 10 epochs.

Loading weights externally
--------------------------

The model weights are located at ``state_dict["state"]["model"]`` within
the stored checkpoint. To load them into a model outside of
a :class:`.Trainer`, use :meth:`torch.load`:

.. code:: python

   model = MyModel()
   state_dict = torch.load("/path/to/checkpoint/ep15.pt")
   model.load_state_dict(state_dict["state"]["model"])

Uploading to Object Store
-------------------------

Checkpoints can also be saved to and loaded from your object store of
choice (e.g. AWS S3 or Google Cloud Storage). Writing checkpoints to an
object store is a two-step process. The checkpoints are first written to
the local filesystem, and then the :class:`RunDirectoryUploader` callback
will upload to the object store.

.. note ::

    We use :mod:`libcloud` to connect to the remote object stores, so be
    sure to have the Python package ``apache-libcloud`` installed.

For this, the :class:`.ObjectStore` needs to be configured with
the following arguments:

-  ``provider``: The name of the object store provider, as recognized by
   :mod:`libcloud`. See available providers
   `here <https://libcloud.readthedocs.io/en/stable/storage/supported_providers.html#provider-matrix>`__.
-  ``container``: The name of the container (i.e. “bucket”) to use.

To prevent accidental leakage of API keys, your secrets must be provided
indirectly through environment variables. Set these in your environment
and provide the following environment variable names:

-  ``key_environ``: The environment variable where your username is
   stored. For example, the GCS access key.
-  ``secret_environ``: The environment variable where your secret is
   stored. For example, the GCS secret that is paired with the
   above access key for requests.

The object store also accepts these common optional arguments:

-  ``host``: The specific hostname for the cloud provider, letting you
   override the default value provided by :mod:`libcloud`.
-  ``port``: The port for the cloud provider
-  ``region``: The region to use for the cloud provider

If your cloud provider requires additional parameters, pass them as a
dictionary under the key ``extra_init_kwargs``.

Once you've configured your object store properly per above, all that's
left is to add the :class:`.RunDirectoryUploader` as a callback.

Let's put all this together below:

.. code:: python

   import uuid
   from composer.callbacks import RunDirectoryUploader
   from composer.utils.object_store import ObjectStoreHparams

   credentials = {"provider": "GOOGLE_STORAGE",
                  "container": "checkpoints-debugging",
                  "key_environ": "GCE_KEY",
                  "secret_environ": "GCE_SECRET"}
   hp = ObjectStoreHparams(**credentials)

   prefix = f"my-model-{str(uuid.uuid4())[:6]}"
   store_uploader = RunDirectoryUploader(hp, object_name_prefix=prefix)

   trainer = Trainer(model=model,
                     train_dataloader=dataloader,
                     eval_dataloader=None,
                     max_duration="90ep",
                     save_folder="checkpoints",
                     callbacks=[store_uploader])

This will train your model, saving the checkpoints locally, and also
upload them to Google Storage buckets using the username from
``GCS_KEY`` and the secrets from ``GCS_SECRET`` in your environment
variables.

Loading from Object Store
-------------------------

Checkpoints saved to an object store can also be loaded in the
same way as files saved on disk. Provide the
:class:`.ObjectStoreHparams` to the trainer's ``load_object_store``
argument.  The ``load_path_format`` argument
should be the path to the checkpoint file *within the container/bucket*.

.. code:: python

   from composer.utils.object_store import ObjectStoreHparams

   credentials = {"provider": "GOOGLE_STORAGE",
                  "container": "checkpoints-debugging",
                  "key_environ": "GCS_KEY",
                  "secret_environ": "GCS_SECRET"}
   hp = ObjectStoreHparams(
       provider="GOOGLE_STORAGE",
       container="checkpoints-debugging",
       key_environ="GCS_KEY",
       secret_environ="GCS_SECRET",
   )
   object_store = hp.initialize_object()

From there we can fine-tune with:

.. code:: python

   new_trainer = Trainer(model=model,
                         train_dataloader=finetune_dataloader,
                         eval_dataloader=None,
                         max_duration="10ep",
                         load_path_format="simple/rank_0/checkpoints/ep1.tar",
                         load_object_store=object_store,
                         load_weights_only=True)
   new_trainer.fit()

Trainer checkpoint API
----------------------

The :class:`.Trainer` has many arguments, and below we provide the API reference
for the arguments that are specific to checkpoint loading and saving:

..
    This is manually duplicated for now, until an automated solution with sphinx
    is implemented.

Loading
~~~~~~~

- ``load_path_format`` (``str``, `optional`): Path to a specific checkpoint to load. If not set (the default),
  then no checkpoint will be loaded. (default: ``None``)
- ``load_object_store`` (:class:`.ObjectStore`, `optional`): For loading from object stores (e.g. S3),
  this will be used to download the checkpoint. Ignored if ``load_path_format`` is not specified. (default: ``None``)
- ``load_weights_only`` (``bool``): Only load the model weights.  Ignored if ``load_path_format`` is not specified.
  (default: ``False``)
- ``load_strict`` (``bool``): Ensure that the set of weights in the checkpoint and model must exactly match. Ignored if
  ``load_path_format`` is not specified. (default: ``False``)
- ``load_chunk_size`` (``int``): Chunk size (in bytes) to use when downloading checkpoints.
  Ignored if the ``load_path_format`` is not specified or it is a local file path. (default: ``1,048,675``)
- ``load_progress_bar`` (``bool``): Display the progress bar for downloading the checkpoint. Ignored if
  ``load_path_format`` is not specified or if it is a local file path. (default: ``True``)

Saving
~~~~~~

- ``save_folder`` (``str``, `optional`): Folder path to save checkpoints, relative to the run directory.
  Set to ``None`` to not save checkpoints. (default: ``None``)
- ``save_interval`` (``str`` or ``int``): How often to save checkpoints. For example, set to "1ep" to save checkpoints
  every epoch, or "10ba" to save checkpoints every 10 batches. An integer will be assumed to be epochs.
  (default: ``1ep``)
- ``save_compression`` (``str``): Compression algorithm to run on checkpoints. Can be ``gzip``, ``bzip2``,
  ``lzma``, or left blank for no compression.  (default: ``""`` for no compression).

Object Store API
----------------

.. autoclass:: composer.utils.object_store.ObjectStoreHparams
    :noindex:

RunDirectoryUploader API
------------------------

.. autoclass:: composer.callbacks.run_directory_uploader.RunDirectoryUploader
    :noindex:

.. TODO: add details on what can be overridden when loading a checkpoint.