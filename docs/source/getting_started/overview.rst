Overview
=======================================================================================================================

**MosaicML** is an open source library designed for modular plug-in of neural network training algorithms on top of PyTorch. With MosaicML, novel **speed-up** and **accuracy-boosting** methods can be seemlessly composed into complete training recipes.


Introduction
-----------------------------------------------------------------------------------------------------------------------
Data scientists and researchers are overwhelmed by the plethora of new algorithms in the literature and open source. Already, several open source platforms have gained popularity, in organizing the space of NLP models (e.g. HuggingFace), or cataloging the zoo of papers and associated code (e.g. Papers with Code). 

Yet, the landscape of algorithms and models is still unorganized, particularly for ML training. It is often difficult to integrate new methods into existing code. In addition, methods are often not well-characterized “in production” nor do they necessarilly **compose** well with other algorithms. 

More generally, it is hard to discern whether particular algorithms accelerate time-to-train, or reduce cost across hardware.

We therefore build MosaicML as an open source library with a standardized way to implement and compose novel algorithms.


Features
-----------------------------------------------------------------------------------------------------------------------
- Standardized way to implement and compose novel algorithms
- Central repository for speed-up and accuracy boosting algorithms
- Composability across a set of popular trainers used in the community today: pytorch lightning, fast.ai, and potentially huggingface and/or deepspeed


Key Concepts
-----------------------------------------------------------------------------------------------------------------------
Let's have a look at MosaicML key concepts:


Supported Algorithms
-----------------------------------------------------------------------------------------------------------------------

What is a neural network algorithm? An algorithm can be anything from a new optimizer (e.g. AdaHessian), a learning rule (e.g. BackDrop), a data augmentation scheme (e.g. MixUp), or an architectural motif (e.g. Blurpool).

The following algorithms are currently supported:

..
    These should be formatted and updated automatically



+-----------------------------+--------------------------+-------------------------+-------------------------+
|            Method           |           Category       |       Needs Custom      |   Reference             |
+=============================+==========================+=========================+=========================+
|           Backdrop          |       Learning Rule      |       ...               |     Here                |
+-----------------------------+--------------------------+-------------------------+-------------------------+
|      Channels Last          |       Example Selection  |      ...                |   Here                  |
+-----------------------------+--------------------------+-------------------------+-------------------------+
|      Mixup                  |       Regularization     |       ...               |   Here                  |
+-----------------------------+--------------------------+-------------------------+-------------------------+
|      Label Smoothing        |       Example Selection  |       ...               |   Here                  |
+-----------------------------+--------------------------+-------------------------+-------------------------+
|      Blurpool               |       Architecture Priors|       Graph Surgery     |   Here                  |
+-----------------------------+--------------------------+-------------------------+-------------------------+


