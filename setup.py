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

def package_files(prefix: str, directory: str, extension: str):
    # from https://stackoverflow.com/a/36693250
    paths = []
    for (path, _, filenames) in os.walk(os.path.join(prefix, directory)):
        for filename in filenames:
            if filename.endswith(extension):
                paths.append(os.path.relpath(os.path.join(path, filename), prefix))
    return paths

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Hide the content between <!-- SETUPTOOLS_LONG_DESCRIPTION_HIDE_BEGIN --> and
# <!-- SETUPTOOLS_LONG_DESCRIPTION_HIDE_END --> tags in the README
while True:
    start_tag = "<!-- SETUPTOOLS_LONG_DESCRIPTION_HIDE_BEGIN -->"
    end_tag = "<!-- SETUPTOOLS_LONG_DESCRIPTION_HIDE_END -->"
    start = long_description.find(start_tag)
    end = long_description.find(end_tag)
    if start == -1:
        assert end == -1, "there should be a balanced number of start and ends"
        break
    else:
        assert end != -1, "there should be a balanced number of start and ends"
        long_description = long_description[:start] + long_description[end + len(end_tag):]

install_requires = [
    "pyyaml>=5.4.1,<6",
    "tqdm>=4.62.3,<5",
    "torchmetrics>=0.7.0,<0.8",
    "torch_optimizer>=0.1.0,<0.2",
    "torchvision>=0.10.0",  # torchvision has strict pytorch requirements
    "torch>=1.9,<2",
    "yahp==0.1.0",
    "requests>=2.26.0,<3",
    "numpy==1.21.5",  # TODO should this be >=1.21.5.10,<2
    "apache-libcloud>=3.3.1,<4",
    "psutil>=5.8.0,<6",
    'coolname>=1.1.0,<2',
]
extra_deps = {}

extra_deps['base'] = []

extra_deps['dev'] = [
    # Imports for docs builds and running tests
    # Pinning versions strictly to avoid random test failures.
    # Should manually update dependency versions occassionally.
    "custom_inherit==2.3.2",
    'junitparser==2.4.3',
    'coverage[toml]==6.3.2',
    'fasteners==0.17.3',  # run_directory_uploader tests require fasteners
    'pytest==7.1.0',
    'toml==0.10.2',
    'yapf==0.32.0',
    'isort==5.10.1',
    'ipython==7.32.0',
    'ipykernel==6.9.2',
    'jupyter==1.0.0',
    'yamllint==1.26.3',
    'pytest-timeout==2.1.0',
    'recommonmark==0.7.1',
    'sphinx==4.4.0',
    # embedding md in rst require docutils>=0.17. See
    # https://myst-parser.readthedocs.io/en/latest/sphinx/use.html?highlight=parser#include-markdown-files-into-an-rst-file
    'docutils==0.17.1',
    'sphinx_markdown_tables==0.0.15',
    'sphinx-argparse==0.3.1',
    'sphinxcontrib.katex==0.8.6',
    'sphinxext.opengraph==0.6.1',
    'sphinxemoji==0.2.0',
    'furo==2022.3.4',
    'sphinx-copybutton==0.5.0',
    'testbook==0.4.2',
    'myst-parser==0.16.1',
    'pylint==2.12.2',
    'docformatter==1.4',
    'sphinx_panels==0.6.0',
    'sphinxcontrib-images==0.9.4',
    # need webdataset to run pyight. Including here to pass pyright.
    # TODO Remove once https://github.com/mosaicml/composer/issues/771 is fixed.
    'webdataset==0.1.103',
]

extra_deps["deepspeed"] = [
    'deepspeed==0.5.10',  # TODO should this be >=0.5.10,<0.6
]

extra_deps["wandb"] = [
    'wandb>=0.12.10,<0.13',
]

extra_deps["unet"] = [
    'monai>=0.8.0,<0.9',
    'scikit-learn>=1.0.1,<2',
]

extra_deps["vit"] = [
    'vit_pytorch==0.27',
]

extra_deps["timm"] = [
    'timm>=0.5.4,<0.6',
]

extra_deps["coco"] = [
    'pycocotools>=2.0.4,<3',
]

extra_deps["nlp"] = [
    'transformers>=4.11,<5',
    'datasets>=1.14,<2',
]

extra_deps['webdataset'] = [
    # PyPI does not permit git dependencies. See https://github.com/mosaicml/composer/issues/771
    # 'webdataset @ git+https://github.com/mosaicml/webdataset.git@dev'
    'wurlitzer>=3.0.2,<4',
]

extra_deps['all'] = set(dep for deps in extra_deps.values() for dep in deps)

composer_data_files = ['py.typed']
composer_data_files += package_files('composer', 'yamls', ".yaml")
composer_data_files += package_files('composer', 'algorithms', ".json")

setup(name="mosaicml",
      version="0.5.0",
      author="MosaicML",
      author_email="team@mosaicml.com",
      description="Composer provides well-engineered implementations of efficient training methods to give "
      "the tools that help you train a better model for cheaper.",
      long_description=long_description,
      long_description_content_type="text/markdown",
      url="https://github.com/mosaicml/composer",
      include_package_data=True,
      package_data={
          "composer": composer_data_files,
      },
      packages=setuptools.find_packages(exclude=["docker*", "notebooks*", "scripts*", "tests*"]),
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
