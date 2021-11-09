# Copyright 2021 MosaicML. All Rights Reserved.

import os
import sys

import setuptools
from setuptools import setup


def package_files(directory):
    # from https://stackoverflow.com/a/36693250
    paths = []
    for (path, directories, filenames) in os.walk(directory):
        for filename in filenames:
            paths.append(os.path.join('..', path, filename))
    return paths


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

install_requires = [
    "pyyaml>=5.4.1",
    "tqdm>=4.62.3",
    "torchmetrics>=0.5.1",
    "torch_optimizer==0.1.0",
    "torchvision>=0.9.0",
    "torch>=1.9",
    "argparse>=1.4.0",
    "yahp>=0.0.10",
]
extra_deps = {}

extra_deps['base'] = []

extra_deps['dev'] = [
    'junitparser>=2.1.1',
    'coverage[toml]>=6.1.1',
    'pytest>=6.2.0',
    'yapf>=0.13.0',
    'isort>=5.9.3',
    'yamllint>=1.26.2',
    'pytest-timeout>=1.4.2',
    'recommonmark>=0.7.1',
    'sphinx>=4.2.0',
    'sphinx_copybutton>=0.4.0',
    'sphinx_markdown_tables>=0.0.15',
    'sphinxcontrib.katex>=0.8.6',
    'sphinxext.opengraph>=0.4.2',
    'sphinx_rtd_theme>=1.0.0',
    'myst-parser>=0.15.2',
]
extra_deps['wandb'] = ['wandb>=0.12.2']

extra_deps['nlp'] = [
    'transformers>=4.11.3',
    'datasets>=1.14.0',
]

extra_deps['unet'] = [
    'medpy==0.4.0',
    'monai>=0.7.0',
    'scikit-learn>=1.0.1',
]

extra_deps['all'] = set(dep for deps in extra_deps.values() for dep in deps)

# TODO fix deferred loading in composer.functional
install_requires.extend(extra_deps['unet'])
install_requires.extend(extra_deps['nlp'])

setup(
    name="mosaicml",
    version="0.2.4",
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
    },
    packages=setuptools.find_packages(include=["composer"]),
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
    install_requires=install_requires,
    extras_require=extra_deps,
    dependency_links=['https://developer.download.nvidia.com/compute/redist'],
    python_requires='>=3.8',
    ext_package="composer",
)

# only visible if user installs with verbose -v flag
# Printing to stdout as not to interfere with setup.py CLI flags (e.g. --version)
print("*" * 20, file=sys.stderr)
print(
    "\nNOTE: For best performance, we recommend installing Pillow-SIMD "
    "\nfor accelerated image processing operations. To install:"
    "\n\n\t pip uninstall pillow && pip install pillow-simd\n",
    file=sys.stderr)
print("*" * 20, file=sys.stderr)
