:orphan:

#################
Dataset/Task Name
#################
.. Based, in part, on [Datasheets for Datasets](https://arxiv.org/pdf/1803.09010.pdf) by Gebru et al.

TLDR
====
..
        TLDR about the dataset name, kind of data, tasks associated with it, and number of examples.

        *E.g., ImageNet is a dataset of natural images for 1000-way classification consisting of 1.2M training examples and 50K validation examples at about resolution 224x224.*

Attribution
===========
**Created By:** _______

.. Who created the dataset (which team, research group) and on behalf of which entity (company, institution, organization)? Provide any links or citations as appropriate.

**License:** Available under _____ license.

.. Under what license is the dataset available?*

Using the Dataset
=================

Obtaining the Dataset
---------------------

The dataset can be obtained from _________.

.. URL, library, or other description of where to get it.

Expected Format
---------------

The Composer library expects this dataset to be stored as ________.

.. What format does the Composer library expect this data to be stored in? E.g. JPEGs in folders for each class, TFRecords with a particular schema, etc.

Steps to Obtain and Prepare Dataset
-----------------------------------

.. A list of step-by-step instructions necessary to obtain this dataset and place it in the right format or a link to such a guide that is available elsewhere.

Performance Considerations:
---------------------------

.. A list of performance considerations in order to use this dataset efficiently. E.g., the cost of loading the data for the first time, recommended storage medium, sensitivity of performance to memory size and disk throughput, costs associated with data augmentation, recommended batching strategies, etc.

Technical Specifications
========================

**Kind of Data: _____**

.. E.g., images, text, etc.

**Content of Each Instance:** ______

.. What information is contained within each instance in the dataset and what source did it come from (if applicable)? (e.g., a sentence from Wikipedia, a picture of a natural object, etc.) If there any labels or other categorical information associated with each instance, mention this as well.

**Specifications of Each Instance:** ________

.. What are the specifications of each example in the dataset? (e.g., a sentence represented as a sequence of words between length 7 and 500, an image of resolution approximately 224x224 with three color channels)

**Number of Training Examples: _____**

**Number of Test Examples: _____**

**Number of Validation Examples: ____**

.. Number of examples in each split of the dataset. Add additional splits as necessary. If an additional holdout set can conditionally be pulled from the training set, mention that too. Even if there is no test or validation set, include the test split above and say "None"

Task(s)
========

.. For each task that is typically associated with this dataset:

**Task: ____**

.. E.g., classification, masked language modeling

**Details:** ____

.. Details associated with the targets of this task. E.g., standard vocabulary size, number of classes.

**Instance Preparation for Training: _____**

.. Details about the standard preparation that is performed on a training instance prior to using it. (e.g., tokenization scheme, normalization, setting to a particular resolution, random cropping)

**Instance Preparation for Inference and Evaluation: _____**

.. Details about the standard preparation that is performed on a test example prior to using it? (see above)

**Metrics: _____**

.. How quality is measured on this task. E.g., perplexity, top-1 accuracy.

**Standard Range of Metrics on Trained Model: ____**

.. The general range of quality that standard approaches get on this task. (Refer to standard approaches as applicable.)

Background
===========

**Original Purpose of Dataset:** _______

.. For what purpose was the dataset originally created? E.g., as a benchmark for a particular task.

**Source of Data: _______**

.. The original source material for the dataset. E.g., 1 million tiny images in the case of CIFAR-10. How was this source data collected.

**Dataset Curation: _____**

.. How were these particular examples chosen? Was any preprocessing applied to these examples?
