# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Configuration file for the Sphinx documentation builder.

This file only contains a selection of the most common options. For a full
list see the documentation:
https://www.sphinx-doc.org/en/master/usage/configuration.html

-- Path setup --------------------------------------------------------------

If extensions (or modules to document with autodoc) are in another directory,
add these directories to sys.path here. If the directory is relative to the
documentation root, use os.path.abspath to make it absolute, like shown here.
"""
import ast
import importlib
import inspect
import json
import os
import shutil
import sys
import tempfile
import types
import warnings
from typing import Any, Dict, List, Tuple, Type

import sphinx.application
import sphinx.ext.autodoc
import sphinx.util.logging
import torch
import torch.nn
from docutils import nodes
from docutils.nodes import Element
from git.repo.base import Repo
from pypandoc.pandoc_download import download_pandoc
from sphinx.ext.autodoc import ClassDocumenter, _
from sphinx.writers.html5 import HTML5Translator

import composer

if not shutil.which('pandoc'):
    # Install pandoc if it is not installed.
    # Pandoc is required by nbconvert but it is not included in the pypandoc pip package
    with tempfile.TemporaryDirectory() as tmpdir:
        # if root on linux, use the "/bin" folder, since "~/bin" = "/root/bin" is not in the path by default
        # similar on osx -- use /Applications instead of "~/Applications" = "/root/Applications"
        target_folder = None
        if os.getuid() == 0:
            if sys.platform == 'linux':
                target_folder = '/bin'
            elif sys.platform == 'darwin':
                target_folder = '/Applications/pandoc'
            # Not handling windows; nobody uses root on windows lol

        download_pandoc(version='2.19.2', download_folder=tmpdir, targetfolder=target_folder, delete_installer=True)

sys.path.insert(0, os.path.abspath('..'))

log = sphinx.util.logging.getLogger(__name__)

# -- Project information -----------------------------------------------------

project = 'MosaicML'
copyright = '2022, MosaicML, Inc.'
author = 'MosaicML'

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.extlinks',
    'sphinx.ext.coverage',
    'sphinx.ext.napoleon',
    'sphinxcontrib.katex',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'sphinxemoji.sphinxemoji',
    'sphinxext.opengraph',
    'sphinx_copybutton',
    'myst_parser',
    'sphinxarg.ext',
    'sphinx.ext.doctest',
    'sphinx_panels',
    'sphinxcontrib.images',
    'nbsphinx',
]


def _get_commit_sha() -> str:
    """Determines the commit sha.

    Returns:
        str: The git commit sha, as a string.
    """
    repo_root = os.path.join(os.path.dirname(__file__), '..', '..')
    repo = Repo(repo_root)
    if repo.is_dirty():
        warning_msg = 'The git repo is dirty. The commit sha for source code links will be incorrect.'
        if os.environ.get('CI', '0') == '0':
            # If developing locally, warn.
            warnings.warn(warning_msg)
        else:
            # If on CI, error.
            raise RuntimeError(warning_msg)
    _commit_sha = repo.commit().hexsha
    return _commit_sha


_COMMIT_SHA = _get_commit_sha()

# Don't show notebook output in the docs
nbsphinx_execute = 'never'

notebook_path = 'mosaicml/composer/blob/' + _COMMIT_SHA + '/{{ env.doc2path(env.docname, base=None) }}'

# Include an "Open in Colab" link at the beginning of all notebooks
nbsphinx_prolog = f"""

.. tip::

    This tutorial is available as a `Jupyter notebook <https://github.com/{notebook_path}>`_.

    ..  image:: https://colab.research.google.com/assets/colab-badge.svg
        :target: https://colab.research.google.com/github/{notebook_path}
        :alt: Open in Colab
"""

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']
source_suffix = ['.rst', '.md']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = [
    '_build', 'Thumbs.db', '.DS_Store', 'examples/imagenet/README.md', 'examples/segmentation/README.md'
]

napoleon_custom_sections = [('Returns', 'params_style')]

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'furo'

# Make sure the target is unique
autosectionlabel_prefix_document = True
autosummary_imported_members = False
autosectionlabel_maxdepth = 5
autosummary_generate = True

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
html_title = ' Composer'

# Customize CSS
html_css_files = ['css/custom.css', 'https://cdn.jsdelivr.net/npm/@docsearch/css@3']
html_js_files = ['js/posthog.js']

# Mosaic logo
# html_logo = 'https://storage.googleapis.com/docs.mosaicml.com/images/logo-dark-bg.png'
html_theme_options = {
    'light_logo': 'logo-light-mode.png',
    'dark_logo': 'logo-dark-mode.png',
    'light_css_variables': {
        'color-brand-primary': '#343434',
        'color-brand-content': '#343434',
    },
    'dark_css_variables': {
        'color-brand-primary': '#f9f9f9',
        'color-brand-content': '#f9f9f9',
    },
}

# Favicon
html_favicon = 'https://mosaic-ml-staging.cdn.prismic.io/mosaic-ml-staging/b1f1a2a0-2b54-4b43-9b76-bfa2e24d6fdf_favicon.svg'

# Don't unfold our common type aliases
autodoc_type_aliases = {
    'Batch': 'composer.core.types.Batch',
}

autodoc_default_options = {
    # don't document the forward() method. Because of how torch.nn.Module.forward is defined in the
    # base class, sphinx does not realize that forward overrides an inherited method.
    'exclude-members': 'hparams_registry'
}
autodoc_inherit_docstrings = False

# Monkeypatch some objects as to exclude their docstrings
torch.nn.Module.forward.__doc__ = ''

torch.nn.Module.forward.__doc__ = None
pygments_style = 'manni'
pygments_dark_style = 'monokai'

html_permalinks = True
html_permalinks_icon = '#'

images_config = {
    'download': False,
}

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'torch': ('https://pytorch.org/docs/stable/', None),
    'torchvision': ('https://pytorch.org/vision/stable/', None),
    'torchtext': ('https://pytorch.org/text/stable/', None),
    'torchmetrics': ('https://torchmetrics.readthedocs.io/en/latest/', None),
    'libcloud': ('https://libcloud.readthedocs.io/en/stable/', None),
    'PIL': ('https://pillow.readthedocs.io/en/stable', None),
    'coolname': ('https://coolname.readthedocs.io/en/latest/', None),
    'datasets': ('https://huggingface.co/docs/datasets/master/en/', None),
    'transformers': ('https://huggingface.co/docs/transformers/master/en/', None),
    'boto3': ('https://boto3.amazonaws.com/v1/documentation/api/latest/', None),
    'botocore': ('https://botocore.amazonaws.com/v1/documentation/api/latest', None),
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
    ('py:class', 'T_nnModule'),
]

python_use_unqualified_type_names = True
autodoc_typehints = 'none'


def skip_redundant_namedtuple_attributes(
    app: sphinx.application.Sphinx,
    what: str,
    name: str,
    obj: Any,
    skip: bool,
    options: sphinx.ext.autodoc.Options,
):
    """Hide the default, duplicate attributes for named tuples."""
    del app, what, name, skip, options
    if '_tuplegetter' in obj.__class__.__name__:
        return True
    return None


with open(os.path.join(os.path.dirname(__file__), 'doctest_fixtures.py'), 'r') as f:
    doctest_global_setup = f.read()

with open(os.path.join(os.path.dirname(__file__), 'doctest_cleanup.py'), 'r') as f:
    doctest_global_cleanup = f.read()


def rstjinja(app, docname, source):
    """Render our pages as a jinja template for fancy templating goodness."""
    # Make sure we're outputting HTML
    if app.builder.format != 'html':
        return
    src = source[0]
    rendered = app.builder.templates.render_string(src, app.config.html_context)
    source[0] = rendered


def get_algorithms_metadata() -> Dict[str, Dict[str, str]]:
    """Get the metadata for algorithms from the ``metadata.json`` files."""
    EXCLUDE = ['no_op_model']

    root = os.path.join(os.path.dirname(__file__), '..', '..', 'composer', 'algorithms')
    algorithms = next(os.walk(root))[1]
    algorithms = [algo for algo in algorithms if algo not in EXCLUDE]

    metadata = {}
    for name in algorithms:
        json_path = os.path.join(root, name, 'metadata.json')

        if os.path.isfile(json_path):
            with open(json_path, 'r') as f:
                data = json.load(f)

            for key, value in data.items():
                if key in metadata:
                    raise ValueError(f'Duplicate keys in metadata: {key}')
                metadata[key] = value

    if not metadata:
        raise ValueError(f'No metadata found, {root} not correctly configured.')
    return metadata


html_context = {'metadata': get_algorithms_metadata()}

# ClassDocumenter.add_directive_header uses ClassDocumenter.add_line to
#   write the class documentation.
# We'll monkeypatch the add_line method and intercept lines that begin
#   with "Bases:".
# In order to minimize the risk of accidentally intercepting a wrong line,
#   we'll apply this patch inside of the add_directive_header method.
# From https://stackoverflow.com/questions/46279030/how-can-i-prevent-sphinx-from-listing-object-as-a-base-class
add_line = ClassDocumenter.add_line
line_to_delete = _('Bases: %s') % u':py:class:`object`'


def _auto_rst_for_module(module: types.ModuleType, exclude_members: List[Any]) -> str:
    """Generate the content of an rst file documenting a module.

    Includes the module docstring, followed by tables for the functions,
    classes, and exceptions

    Args:
        module: The module object to document
        exclude_members: A list of Python objects to exclude from the
            documentation. Providing objects that are not imported in
            ``module`` are ignored.

    Returns:
        The rst content for the module
    """
    name = module.__name__
    lines = []

    functions: List[Tuple[str, types.FunctionType]] = []
    exceptions: List[Tuple[str, Type[BaseException]]] = []
    classes: List[Tuple[str, Type[object]]] = []
    methods: List[Tuple[str, types.MethodType]] = []
    attributes: List[Tuple[str, object]] = []

    # add title and module docstring
    lines.append(f'{name}')
    lines.append(f'{"=" * len(name)}\n')
    lines.append(f'.. automodule:: {name}\n')

    # set prefix so that we can use short names in the autosummaries
    lines.append(f'.. currentmodule:: {name}')

    try:
        all_members = list(module.__all__)
    except AttributeError:
        all_members = list(vars(module).keys())

    for item_name, val in vars(module).items():
        if val in exclude_members:
            continue

        if item_name.startswith('_'):
            # Skip private members
            continue

        if item_name not in all_members:
            # Skip members not in `__all__``
            continue

        if isinstance(val, types.ModuleType):
            # Skip modules; those are documented by autosummary
            continue

        if isinstance(val, types.FunctionType):
            functions.append((item_name, val))
        elif isinstance(val, types.MethodType):
            methods.append((item_name, val))
        elif isinstance(val, type) and issubclass(val, BaseException):
            exceptions.append((item_name, val))
        elif isinstance(val, type):
            assert issubclass(val, object)
            classes.append((item_name, val))
        else:
            attributes.append((item_name, val))
            continue

    # Sort by the reimported name
    functions.sort(key=lambda x: x[0])
    exceptions.sort(key=lambda x: x[0])
    classes.sort(key=lambda x: x[0])
    attributes.sort(key=lambda x: x[0])

    for category, category_name in ((functions, 'Functions'), (classes, 'Classes'), (exceptions, 'Exceptions')):
        sphinx_lines = []
        for item_name, _ in category:
            sphinx_lines.append(f'      {item_name}')
        if len(sphinx_lines) > 0:
            lines.append(f'\n.. rubric:: {category_name}\n')
            lines.append('.. autosummary::')
            lines.append('      :toctree: generated')
            lines.append('      :nosignatures:')
            if category_name in ('Classes', 'Hparams'):
                lines.append('      :template: classtemplate.rst')
            elif category_name == 'Functions':
                lines.append('      :template: functemplate.rst')
            lines.append('')
            lines.extend(sphinx_lines)
            lines.append('')

    lines.append('.. This file autogenerated by docs/source/conf.py\n')

    return '\n'.join(lines)


def _modules_to_rst() -> List[types.ModuleType]:
    """Return the list of modules for which to generate API reference rst files."""
    # adding composer.functional to the below list yields:
    #   AttributeError: module 'composer' has no attribute 'functional'
    import composer.functional as cf

    document_modules: List[types.Module] = [
        composer,
        cf,
        composer.utils.dist,
        composer.utils.reproducibility,
        composer.core.types,
    ]
    exclude_modules: List[types.Module] = [composer.trainer, composer._version]
    for name in composer.__dict__:
        obj = composer.__dict__[name]
        if isinstance(obj, types.ModuleType) and obj not in exclude_modules:
            document_modules.append(obj)

    return document_modules


def _generate_rst_files_for_modules() -> None:
    """Generate .rst files for each module to include in the API reference.

    These files contain the module docstring followed by tables listing all
    the functions, classes, etc.
    """
    docs_dir = os.path.abspath(os.path.dirname(__file__))
    module_rst_save_dir = os.path.join(docs_dir, 'api_reference')
    # gather up modules to generate rst files for
    document_modules = _modules_to_rst()

    # rip out types that are duplicated in top-level composer module
    composer_imported_types = []
    for name in composer.__all__:
        obj = composer.__dict__[name]
        if not isinstance(obj, types.ModuleType):
            composer_imported_types.append(obj)

    document_modules = sorted(document_modules, key=lambda x: x.__name__)
    os.makedirs(module_rst_save_dir, exist_ok=True)
    for module in document_modules:
        saveas = os.path.join(module_rst_save_dir, module.__name__ + '.rst')
        print(f'Generating rst file {saveas} for module: {module.__name__}')

        # avoid duplicate entries in docs. We add torch's _LRScheduler to
        # types, so we get a ``WARNING: duplicate object description`` if we
        # don't exclude it
        exclude_members = [torch.optim.lr_scheduler._LRScheduler]
        if module is not composer:
            exclude_members += composer_imported_types

        content = _auto_rst_for_module(module, exclude_members=exclude_members)

        with open(saveas, 'w') as f:
            f.write(content)


def _add_line_no_object_base(self, text, *args, **kwargs):
    if text.strip() == line_to_delete:
        return

    add_line(self, text, *args, **kwargs)


add_directive_header = ClassDocumenter.add_directive_header


def _add_directive_header_no_object_base(self, *args, **kwargs):
    """Hide that all classes inherit from the base class ``object``."""
    self.add_line = _add_line_no_object_base.__get__(self)

    result = add_directive_header(self, *args, **kwargs)

    del self.add_line

    return result


ClassDocumenter.add_directive_header = _add_directive_header_no_object_base


def _recursive_getattr(obj: Any, path: str):
    parts = path.split('.')
    try:
        obj = getattr(obj, parts[0])
    except AttributeError:
        return None
    path = '.'.join(parts[1:])
    if path == '':
        return obj
    else:
        return _recursive_getattr(obj, path)


def _determine_lineno_of_attribute(module: types.ModuleType, attribute: str):
    # inspect.getsource() does not work with module-level attributes
    # instead, parse the module manually using ast, and determine where
    # the expression was defined
    source = inspect.getsource(module)
    filename = inspect.getsourcefile(module)
    assert filename is not None, f'filename for module {module} could not be found'
    ast_tree = ast.parse(source, filename)
    for stmt in ast_tree.body:
        if isinstance(stmt, ast.Assign):
            if any(isinstance(x, ast.Name) and x.id == attribute for x in stmt.targets):
                return stmt.lineno
    return None


class PatchedHTMLTranslator(HTML5Translator):
    """Open all external links in a new tab."""

    # Adapted from https://stackoverflow.com/a/61669375

    def visit_reference(self, node: Element) -> None:
        atts = {'class': 'reference'}
        if node.get('internal') or 'refuri' not in node:
            atts['class'] += ' internal'
        else:
            atts['class'] += ' external'
            # ---------------------------------------------------------
            # Customize behavior (open in new tab, secure linking site)
            if 'refid' not in node and (not any(node['refuri'].startswith(x)
                                                for x in ('/', 'https://docs.mosaicml.com', '#'))):
                # If there's a refid, or the refuri starts with a non-external uri scheme, then it's an internal
                # (hardcoded) link, so don't open that in a new tab
                # Otherwise, it's really an external link. Open it in a new tab.
                atts['target'] = '_blank'
                atts['rel'] = 'noopener noreferrer'
            # ---------------------------------------------------------
        if 'refuri' in node:
            atts['href'] = node['refuri'] or '#'
            if self.settings.cloak_email_addresses and atts['href'].startswith('mailto:'):
                atts['href'] = self.cloak_mailto(atts['href'])
                self.in_mailto = True
        else:
            assert 'refid' in node, \
                   'References must have "refuri" or "refid" attribute.'
            atts['href'] = '#' + node['refid']
        if not isinstance(node.parent, nodes.TextElement):
            assert len(node) == 1 and isinstance(node[0], nodes.image)
            atts['class'] += ' image-reference'
        if 'reftitle' in node:
            atts['title'] = node['reftitle']
        if 'target' in node:
            atts['target'] = node['target']
        self.body.append(self.starttag(node, 'a', '', **atts))

        if node.get('secnumber'):
            self.body.append(('%s' + self.secnumber_suffix) % '.'.join(map(str, node['secnumber'])))


def setup(app: sphinx.application.Sphinx):
    """Setup hook."""
    _generate_rst_files_for_modules()
    app.connect('autodoc-skip-member', skip_redundant_namedtuple_attributes)
    app.connect('source-read', rstjinja)
    app.set_translator('html', PatchedHTMLTranslator)
