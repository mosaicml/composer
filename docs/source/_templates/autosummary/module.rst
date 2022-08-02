.. From https://github.com/sphinx-doc/sphinx/tree/4.x/sphinx/ext/autosummary/templates/autosummary/module.rst

{{ name | escape | underline}}

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
   __module__ should be manually overridden -- i.e. in the `__init__.py` which contains `from xxx import YYY`,
   add in `YYY.__module__ = __name__`.

.. automodule:: {{ fullname }}
   :members:
   :show-inheritance:
   :ignore-module-all:
