|:robot:| Algorithms
====================

Under construction |:construction:|

.. csv-table::
    :header: "Name" , "Summary"
    :delim: |
    :widths: 30, 70

    {% for name, data in metadata.items() %}
    :doc:`{{ data.class_name }}</method_cards/{{name}}>` | {{ data.summary }}
    {% endfor %}
