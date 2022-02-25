|:robot:| Algorithms
====================

Under construction |:construction:|

Included in the Composer library is a suite of algorithmic speedup algorithms.
These modify the basic training procedure, and are intended to be *composed*
together to easily create a complex and hopefully more efficient training routine.
While other libraries may have implementations of some of these, the implementations
in Composer are specifically written to be combined with other methods.

Below is a brief overview of the [algorithms currently available in Composer](https://github.com/mosaicml/composer/tree/dev/composer/algorithms).
For more detailed information about each algorithm, see the method cards,
also linked in the table. Each algorithm has a functional implementation intended
for use with your own training loop, and an implementation intended for use with
Composer's trainer.

.. csv-table::
    :header: "Name" , "tldr", "functional"
    :delim: |
    :widths: 30, 40, 30

    {% for name, data in metadata.items() %}
    :doc:`{{ data.class_name }}</method_cards/{{name}}>` | {{ data.tldr }} | ``{{ data.functional }}``
    {% endfor %}
