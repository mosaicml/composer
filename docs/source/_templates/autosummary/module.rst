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

   We use :ignore-module-all: so sphinx does not document the same module twice, even if it is reimported
   For reimports that should be documented somewhere other than where they are defined, the re-imports
   __module__ should be manually overridden

.. automodule:: {{ fullname }}
   :members:
   :ignore-module-all:
