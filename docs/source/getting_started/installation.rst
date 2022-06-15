|:floppy_disk:| Installation
============================

``Composer`` is available with pip:

.. code-block::

    pip install mosaicml

as well as with Anaconda:

.. code-block::

    conda install -c mosaicml mosaicml

To include non-core dependencies that are required by some algorithms, callbacks, datasets, or models,
the following installation targets are available:

* ``pip install 'mosaicml[dev]'``: Installs development dependencies, which are required for running tests
  and building documentation.
* ``pip install 'mosaicml[deepspeed]'``: Installs Composer with support for :mod:`deepspeed`.
* ``pip install 'mosaicml[nlp]'``: Installs Composer with support for NLP models and algorithms.
* ``pip install 'mosaicml[unet]'``: Installs Composer with support for :doc:`Unet </model_cards/unet>`.
* ``pip install 'mosaicml[timm]'``: Installs Composer with support for :mod:`timm`.
* ``pip install 'mosaicml[wandb]'``: Installs Composer with support for :mod:`wandb`.
* ``pip install 'mosaicml[all]'``: Install all optional dependencies.

For a developer install, clone directly:

.. code-block::

    git clone https://github.com/mosaicml/composer.git
    cd composer
    pip install -e ".[all]"


.. note::

    For fast loading of image data, we **highly** recommend installing
    `Pillow-SIMD <https://github.com/uploadcare/pillow-simd>`_\.  To install, vanilla pillow must first be uninstalled.

    .. code-block::

        pip uninstall pillow && pip install pillow-simd

    Pillow-SIMD is not supported for Apple M1 Macs.


.. include:: ../../../docker/README.md
   :parser: myst_parser.sphinx_
