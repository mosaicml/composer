# Copyright 2021 MosaicML. All Rights Reserved.

import os
import site
import sys
import textwrap

import setuptools
from setuptools import setup
from setuptools.command.develop import develop as develop_orig

_IS_ROOT = os.getuid() == 0
_IS_USER = "--user" in sys.argv[1:]
_IS_VIRTUALENV = "VIRTUAL_ENV" in os.environ


# From https://stackoverflow.com/questions/51292333/how-to-tell-from-setup-py-if-the-module-is-being-installed-in-editable-mode
class develop(develop_orig):

    def run(self):
        if _IS_ROOT and (not _IS_VIRTUALENV) and (not _IS_USER):
            raise RuntimeError(
                textwrap.dedent("""\
                    When installing in editable mode as root outside of a virtual environment,
                    please specify `--user`. Editable installs as the root user outside of a virtual environment
                    do not work without the `--user` flag. Please instead run something like: `pip install --user -e .`"""
                               ))
        super().run()


# From https://github.com/pypa/pip/issues/7953#issuecomment-645133255
site.ENABLE_USER_SITE = _IS_USER


def package_files(directory: str):
    # from https://stackoverflow.com/a/36693250
    paths = []
    for (path, _, filenames) in os.walk(directory):
        for filename in filenames:
            paths.append(os.path.join('..', path, filename))
    return paths


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

install_requires = [
    "pyyaml>=5.4.1",
    "tqdm>=4.62.3",
    "torchmetrics>=0.6.0",
    "torch_optimizer==0.1.0",
    "torchvision>=0.9.0",
    "torch>=1.9",
    "yahp>=0.0.14",
    "requests>=2.26.0",
    "numpy==1.21.5",
    "apache-libcloud>=3.3.1",
    "psutil>=5.8.0",
]
extra_deps = {}

extra_deps['base'] = []

extra_deps['dev'] = [
    # Imports for docs builds and running tests
    "custom_inherit==2.3.2",
    'junitparser>=2.1.1',
    'coverage[toml]>=6.1.1',
    'fasteners>=0.16.3',  # run_directory_uploader tests require fasteners
    'pytest>=7.0.0',
    'toml>=0.10.2',
    'yapf>=0.32.0',
    'isort>=5.9.3',
    'ipython>=7.29.0',
    'ipykernel>=6.5.0',
    'jupyter>=1.0.0',
    'yamllint>=1.26.2',
    'pytest-timeout>=1.4.2',
    'pyright==1.1.224.post1',
    'recommonmark>=0.7.1',
    'sphinx>=4.2.0',
    'sphinx_copybutton>=0.4.0',
    'sphinx_markdown_tables>=0.0.15',
    'sphinx-argparse>=0.3.1',
    'sphinxcontrib.katex>=0.8.6',
    'sphinxext.opengraph>=0.4.2',
    'sphinxemoji>=0.2.0',
    'furo>=2022.1.2',
    'sphinx-copybutton>=0.4.0',
    'testbook>=0.4.2',
    'myst-parser==0.16.1',
    'pylint>=2.12.2',
    'docformatter>=1.4',
    'sphinx_panels>=0.6.0',
]

extra_deps["deepspeed"] = [
    'deepspeed>=0.5.5',
]

extra_deps["wandb"] = [
    'wandb>=0.12.2',
]

extra_deps["unet"] = [
    'monai>=0.7.0',
    'scikit-learn>=1.0.1',
]

extra_deps["timm"] = [
    'timm>=0.5.4',
]

extra_deps["nlp"] = [
    'transformers>=4.11',
    'datasets>=1.14',
]

extra_deps['all'] = set(dep for deps in extra_deps.values() for dep in deps)

setup(name="mosaicml",
      version="0.4.0",
      author="MosaicML",
      author_email="team@mosaicml.com",
      description="composing methods for ML training efficiency",
      long_description=long_description,
      long_description_content_type="text/markdown",
      url="https://github.com/mosaicml/composer",
      include_package_data=True,
      package_data={
          "composer": ['py.typed'],
          "": package_files('composer/yamls'),
          "": package_files('composer/algorithms')
      },
      packages=setuptools.find_packages(exclude=["tests*"]),
      classifiers=[
          "Programming Language :: Python :: 3",
          "Programming Language :: Python :: 3.7",
          "Programming Language :: Python :: 3.8",
          "Programming Language :: Python :: 3.9",
      ],
      install_requires=install_requires,
      entry_points={
          'console_scripts': ['composer = composer.cli.launcher:main',],
      },
      extras_require=extra_deps,
      dependency_links=['https://developer.download.nvidia.com/compute/redist'],
      python_requires='>=3.7',
      ext_package="composer",
      cmdclass={'develop': develop})

# only visible if user installs with verbose -v flag
# Printing to stdout as not to interfere with setup.py CLI flags (e.g. --version)
print("*" * 20, file=sys.stderr)
print(textwrap.dedent("""\
    NOTE: For best performance, we recommend installing Pillow-SIMD
    for accelerated image processing operations. To install:
    \t pip uninstall pillow && pip install pillow-simd"""),
      file=sys.stderr)
print("*" * 20, file=sys.stderr)
