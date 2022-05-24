:orphan:

############
Method Name
############
.. Name goes here

AKA
====
.. Other names for the method found in the literature.

Idea for a snazzy graphic or schematic
=======================================
.. Describe a potential snazzy graphic/schematic here or link to one from a paper. Not sure if this will make the cut, but put down ideas or sketches here so that we can have graphic designers create one or use one from the papers.

Attribution
============
.. Links to papers or code with names of authors.  Here's how to do an external link:

Davis W. Blalock and John V. Guttag, `Multiplying Matrices Without Multiplying <https://arxiv.org/abs/2106.10860>`_

Code
=====
.. Link to code in Composer
.. Can also add a code block like this:
.. code:: python

        def some_function():
                msg = "Hello World"
                print(msg)

Method or Best Practice?
=========================

TLDR
=====
.. A brief TLDR of how the method works, what it applies to, and what it does in a single sentence. "ColOut removes entire rows or columns from inputs to vision models to regularize and reduce the amount of computation."

Tags for Method
================
.. We'll have to create high-level tags like "speedup," "regularization," "curriculum," "augmentation," etc.

Applicable Settings
====================
.. Models, tasks, and settings (e.g., ResNet-50 on ImageNet, computer vision, language modeling) where this is applicable.

..
        Please use the terms from the `categories of tasks <https://www.notion.so/1a4b8d2e45f04088ae83c61001fa8d48>`_,
        `kinds of tasks <https://www.notion.so/9ce3870f659c4428b1cdb43476eee23a>`_, `tasks <https://www.notion.so/6f7799d6cacb4db984d80856e536d11a>`_,
        `network families <https://www.notion.so/12eb15bf98fd469d817ce095703aa54d>`_, `networks <https://www.notion.so/2951ae64641d435db3fe6ca3dd6b3f89>`_,
        and `settings <https://www.notion.so/efb6ac1452da4333b9bc439eb18d6bf7>`_ as applicable. Think of these as tags that we might eventually want
        to allow people to filter for.

High-Level Summary
===================
.. A high-level overview of how the technique works. One or two sentences max.

Example Effects
================
.. Examples of speed and accuracy changes (and other effects) induced by this technique in exemplary settings on its own.

Implementation Details
=======================
.. More detailed considerations for making this technique work properly. E.g., the second forward pass in Backdrop.  Examples of some markup:

Might want to include some inline math like this: :math:`a^2 + b^2 = c^2`.

Or even fancier: :math:`f(x)=\int_{a}^{b}x^2dx`

Can also just do a math directive like this:

.. math::
        (a + b)^2 = a^2 + 2ab + b^2

Suggested Hyperparameters
==========================
.. Details on hyperparameters that work well.  Here are some table examples if you want to use that here:
.. list-table:: List Table Example
        :widths: 20 20 20
        :header-rows: 1

        * - Hyperparameters
          - Values
          - Suggested Value
        * - alpha
          - 0 - :math:`\infty`
          - 0.2
        * - beta
          - 0 - 10
          - 2

.. csv-table:: CSV Table Example
        :header: "Hyperparameters", "Values", "Suggested Value"
        :widths: 20, 20, 20

        "alpha", "0 - :math:`\infty`", 0.2
        "beta", "0 - 10", 2

Considerations
===============
.. Tradeoffs for when to use this method. E.g., CPU demands of randaugment, or the overhead associated with trying to extract performance benefits from stochcastic depth. Ideally, this section will include graphs to illustrate these points. This could include settings where the method works well or poorly, hardware where it works well or poorly, etc.

Composability
==============
.. Considerations for when this method will/won't get along well with other methods. E.g., too much regularization or too much focus on speeding up one part of the pipeline.

Side-Effects
=============
.. Reasons that this method could lead to unintended consequences.

Detailed Results
=================
.. Tell the full story of the method here, experiments conducted, hyperparameters, etc. How we came to the decisions that have been described above.
