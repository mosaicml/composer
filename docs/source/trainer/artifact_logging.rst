|:hindu_temple:| Artifact Logging
====================

What is an artifact?
-----------

An artifact can be thought of as a folder stored in the cloud. This folder can contain any
files or file URL that are pertinent to a training run. These files are called **assets**
or **asset files**. Assets can be any type of inputs to or byproducts of a training run,
like datasets, checkpoints, config files, etc. This folder different from a normal folder
or directory you would have on your local system is that it is ‘versioned’ and it is often
stored in the cloud. It is  versioned in the sense that it will track the whole history of
versions of these asset  files much like `git` or `subversion` tracks code files.
Moreover, this folder is usually stored in the cloud on S3 or GCP, for example.

Artifacts and Assets in Composer
-----------

Artifact amd asset creation and artifact uploading/logging in Composer two separate 
operations that must be separately configured for. For example, it is possible to set up
checkpoint logging or metric logging in Composer, but if you don’t configure artifact
logging, these files will not get uploaded to the cloud. Below we will explain 
**asset creation**, **artifact creation** and **artifact logging** in Composer.

Creating Artifacts and Assets in Composer
-----------

When using Composer, most asset files are usually created for you without you even 
realizing it. For example, configuring any of the following will create asset files:

* :doc:`Checkpoint Saving<checkpointing>` will create checkpoint asset files.
* Many :doc:`metric loggers<logging>` will create assets. For example:
    * :class:`~.TensorboardLogger` will create Tensorboard event log asset files.
    * :class:`~.FileLogger` will create log asset files.

* Using our example entrypoint `run_composer_trainer.py` will log a yaml file containing
all of a run's hyperparameters and configurations called `hparams.yaml`

Logging (or Uploading) Artifacts
-----------

Uploading your artifacts (and the assets within your artifacts) to S3 or some
other cloud storage location, like GCP, is what we call **artifact logging** (or 
artifact uploading). To make sure these artifacts are uploaded to cloud storage, you need
to specify a "backend" logger for logging artifacts. Currently we support two 
"backend loggers": :class:`~.ObjectStoreLogger` and :class:`~.WandBLogger`. In order to have your artifacts and their assets
uploaded to the cloud You must
instantiate one of those two loggers . The :class:`~.ObjectStoreLogger` will upload your asset files to
cloud storage, like AWS S3 or GCP. The :class:`~.WandBLogger` will upload your assets to
your WandB run. See [here](https://docs.wandb.ai/guides/artifacts) for more details on
WandB artifact logging.

Using the ObjectStoreLogger
-----------
There are two key arguments you need to specify to create an :class:`~.ObjectStoreLogger`:

* `object_store_cls`
* `object_store_kwargs`


Putting it All Together: Saving Checkpoints, Logging Metrics, and Uploading them to S3.
-----------
Here is 
