|:robot:| Algorithms
====================

Under construction |:construction:|

.. csv-table::
    :header: "Name" , "tldr", "functional"
    :delim: |
    :widths: 30, 40, 30

    {% for name, data in metadata.items() %}
    :doc:`{{ data.class_name }}</method_cards/{{name}}>` | {{ data.tldr }} | ``{{ data.functional }}``
    {% endfor %}
