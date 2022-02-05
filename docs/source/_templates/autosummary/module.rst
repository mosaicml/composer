.. From https://github.com/sphinx-doc/sphinx/tree/4.x/sphinx/ext/autosummary/templates/autosummary/module.rst

{{ fullname | escape | underline}}

.. List the submodules

{% block modules %}
{% if modules %}
.. rubric:: Modules

.. autosummary::
   :toctree:
   :recursive:
{% for item in modules %}
   {{ item }}
{%- endfor %}
{% endif %}
{% endblock %}

.. Autodoc anything defined in the module itself
.. automodule:: {{ fullname }}
   :members:
