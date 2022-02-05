# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
import types
from typing import List

import yahp as hp

sys.path.insert(0, os.path.abspath('..'))

# -- Project information -----------------------------------------------------

project = 'MosaicML'
copyright = '2022, MosaicML, Inc.'
author = 'MosaicML'

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.extlinks",
    "sphinx.ext.coverage",
    "sphinx.ext.napoleon",
    'sphinxcontrib.katex',
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    'sphinxemoji.sphinxemoji',
    "sphinxext.opengraph",
    "sphinx_copybutton",
    "sphinx_rtd_theme",
    "myst_parser",
    "sphinx.ext.intersphinx",
    "sphinxarg.ext",
    'autodocsumm',
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']
source_suffix = ['.rst', '.md']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

napoleon_custom_sections = [('Returns', 'params_style')]

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#

html_theme = 'sphinx_rtd_theme'

# html_theme = 'sphinx_rtd_theme'

html_theme_options = {
    # Toc options
    'collapse_navigation': False,
    'display_version': False,
    'navigation_depth': 5,
    'logo_only': True,
    'sticky_navigation': False,
    'globaltoc_collapse': True,
    'globaltoc_maxdepth': -1,
}

# Make sure the target is unique
autosectionlabel_prefix_document = True
autosummary_imported_members = False
autosectionlabel_maxdepth = 5
autosummary_generate = True

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# Customize CSS
html_css_files = ['css/custom.css']

# Mosaic logo
html_logo = 'https://storage.googleapis.com/docs.mosaicml.com/images/logo-dark-bg.png'

# Favicon
html_favicon = 'https://mosaic-ml-staging.cdn.prismic.io/mosaic-ml-staging/b1f1a2a0-2b54-4b43-9b76-bfa2e24d6fdf_favicon.svg'

# Don't unfold our common type aliases
autodoc_type_aliases = {
    'Tensor': 'composer.core.types.Tensor',
    'Tensors': 'composer.core.types.Tensors',
    'Batch': 'composer.core.types.Batch',
    'BatchPair': 'composer.core.types.BatchPair',
    'BatchDict': 'composer.core.types.BatchDict',
    'StateDict': 'composer.core.types.StateDict',
    'TDeviceTransformFn': 'composer.core.types.TDeviceTransformFn',
    'Hparams': 'yahp.hparams.Hparams',
}

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'torch': ('https://pytorch.org/docs/stable/', None),
    'yapf': ('https://docs.mosaicml.com/projects/yahp/en/stable/', None),
    'torchvision': ('https://pytorch.org/vision/stable/', None),
    'torchtext': ('https://pytorch.org/text/stable/', None),
    'torchmetrics': ('https://torchmetrics.readthedocs.io/en/latest/', None),
    'libcloud': ('https://libcloud.readthedocs.io/en/stable/', None),
}

nitpicky = False  # warn on broken links

nitpick_ignore = [
    ('py:class', 'type'),
    ('py:class', 'optional'),
    ('py:meth', 'wandb.init'),
    ('py:attr', 'wandb.run.config'),
    ('py:attr', 'wandb.run.tags'),
    ('py:meth', 'torch.save'),
    ('py:meth', 'torch.load'),
    ('py:class', 'TLogDataValue'),
    ('py:class', 'TLogData'),
    ('py:class', 'T_nnModule'),
]

python_use_unqualified_type_names = True
autodoc_inherit_docstrings = True

# monkeypatch hparams docs so we don't get hparams_registry docstrings everywhere
hp.Hparams.__doc__ = ""
hp.Hparams.initialize_object.__doc__ = ""


def maybe_skip_member(app, what: str, name: str, obj, skip: bool, options):
    # Hide the default, duplicate attirubtes for named tuples
    if '_tuplegetter' in obj.__class__.__name__:
        return True
    return None

def add_module_summary_tables(app, what: str, name: str, obj, options, lines: List[str]):
    """This hook adds in summary tables for each module, documenting all functions, exceptions, classes, and attributes.

    It links reimported imports to their original source, as not to create a duplicate, indexed toctree entry.
    It automatically inserts itself at the end of each module docstring. 
    """
    functions = []
    exceptions = []
    classes = []
    attributes = []
    if len(lines) == 0:
        # insert a stub docstring so it doesn't start with functions/exceptions/classes/attributes
        lines.append(obj.__name__)
    if what == "module":

        for name, val in vars(obj).items():
            if name.startswith("_"):
                continue
            if not hasattr(val, "__module__"):
                continue
            if not hasattr(val, "__name__"):
                continue

            if isinstance(val, types.FunctionType):
                functions.append(val)
            elif isinstance(val, type) and issubclass(val, BaseException):
                exceptions.append(val)
            elif isinstance(val, type) and issubclass(val, object):
                classes.append(val)
            elif isinstance(val, object) and not isinstance(val, types.ModuleType):
                attributes.append(val)

        for category, name in ((functions, "Functions"), (classes, "Classes"), (attributes, "Attributes"),
                               (exceptions, "Exceptions")):
            if len(category) > 0:
                lines.append("")
                lines.append(f".. rubric:: {name}")
                lines.append("")
                lines.append(".. autosummary::")
                lines.append("    :nosignatures:")
                lines.append("")
                for item in category:
                    lines.append(f"    ~{item.__module__}.{item.__name__}")


def setup(app):
    app.connect('autodoc-skip-member', maybe_skip_member)
    app.connect('autodoc-process-docstring', add_module_summary_tables)
