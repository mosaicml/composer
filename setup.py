# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Composer package setup."""

import os
import site
import sys
import textwrap

import setuptools
from setuptools import setup
from setuptools.command.develop import develop as develop_orig

# Read the composer version
# Cannot import from `composer.__version__` since that will not be available when building or installing the package
with open(os.path.join(os.path.dirname(__file__), 'composer', '_version.py')) as f:
    version_globals = {}
    version_locals = {}
    exec(f.read(), version_globals, version_locals)
    composer_version = version_locals['__version__']

_IS_ROOT = os.getuid() == 0
_IS_USER = '--user' in sys.argv[1:]
_IS_VIRTUALENV = 'VIRTUAL_ENV' in os.environ


# From https://stackoverflow.com/questions/51292333/how-to-tell-from-setup-py-if-the-module-is-being-installed-in-editable-mode
class develop(develop_orig):
    """Override the ``develop`` class to error if attempting an editable install as root."""

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
    """Get all the files to package."""
    # from https://stackoverflow.com/a/36693250
    paths = []
    for (path, _, filenames) in os.walk(os.path.join(prefix, directory)):
        for filename in filenames:
            if filename.endswith(extension):
                paths.append(os.path.relpath(os.path.join(path, filename), prefix))
    return paths


with open('README.md', 'r', encoding='utf-8') as fh:
    long_description = fh.read()

# Hide the content between <!-- SETUPTOOLS_LONG_DESCRIPTION_HIDE_BEGIN --> and
# <!-- SETUPTOOLS_LONG_DESCRIPTION_HIDE_END --> tags in the README
while True:
    start_tag = '<!-- SETUPTOOLS_LONG_DESCRIPTION_HIDE_BEGIN -->'
    end_tag = '<!-- SETUPTOOLS_LONG_DESCRIPTION_HIDE_END -->'
    start = long_description.find(start_tag)
    end = long_description.find(end_tag)
    if start == -1:
        assert end == -1, 'there should be a balanced number of start and ends'
        break
    else:
        assert end != -1, 'there should be a balanced number of start and ends'
        long_description = long_description[:start] + long_description[end + len(end_tag):]

install_requires = [
    'pyyaml>=6.0,<7',
    'tqdm>=4.62.3,<5',
    'torchmetrics>=0.10.0,<1.1',
    'torch_optimizer>=0.3.0,<0.4',
    'torchvision>=0.13.1,<0.17',
    'torch>=1.13.1,<2.1.1',
    'requests>=2.26.0,<3',
    'numpy>=1.21.5,<1.27.0',
    'psutil>=5.8.0,<6',
    'coolname>=1.1.0,<3',
    'tabulate==0.9.0',  # for auto-generating tables
    'py-cpuinfo>=8.0.0,<10',
    'packaging>=21.3.0,<23',
    'importlib-metadata>=5.0.0,<7',
    'mosaicml-cli>=0.5.8,<0.6',
]
extra_deps = {}

extra_deps['base'] = []

extra_deps['dev'] = [
    # Imports for docs builds and running tests
    # Pinning versions strictly to avoid random test failures.
    # Should manually update dependency versions occassionally.
    'custom_inherit==2.4.1',
    'junitparser==3.1.0',
    'coverage[toml]==7.3.0',
    'fasteners==0.18',  # object store tests require fasteners
    'pytest==7.4.3',
    'toml==0.10.2',
    'ipython==8.11.0',
    'ipykernel==6.26.0',
    'jupyter==1.0.0',
    'yamllint==1.33.0',
    'recommonmark==0.7.1',
    'sphinx==4.4.0',
    'pre-commit>=3.4.0,<4',
    # embedding md in rst require docutils>=0.17. See
    # https://myst-parser.readthedocs.io/en/latest/sphinx/use.html?highlight=parser#include-markdown-files-into-an-rst-file
    'docutils==0.17.1',
    'sphinx_markdown_tables==0.0.17',
    'sphinx-argparse==0.4.0',
    'sphinxcontrib.katex==0.9.6',
    'sphinxext.opengraph==0.9.0',
    'sphinxemoji==0.2.0',
    'furo==2022.9.29',
    'sphinx-copybutton==0.5.2',
    'testbook==0.4.2',
    'myst-parser==0.16.1',
    'sphinx_panels==0.6.0',
    'sphinxcontrib-images==0.9.4',
    'pytest_codeblocks==0.17.0',
    'traitlets==5.13.0',
    'nbsphinx==0.9.1',
    'pandoc==2.3',
    'pypandoc==1.12',
    'GitPython==3.1.40',
    'moto[s3]>=4.0.1,<5',
    'mock-ssh-server==0.9.1',
    'cryptography==41.0.5',
    'pytest-httpserver>=1.0.4,<1.1',
    'setuptools<=59.5.0',
]

extra_deps['health_checker'] = {
    'pynvml>=11.5.0,<12',
}

extra_deps['system_metrics_monitor'] = {
    'pynvml>=11.5.0,<12',
}

extra_deps['slack'] = {
    'slack_sdk>=3.19.5,<4',
}

extra_deps['deepspeed'] = [
    'deepspeed==0.8.3',
    'pydantic>=1.0,<2',
]

extra_deps['wandb'] = [
    'wandb>=0.13.2,<0.17',
]

extra_deps['comet_ml'] = [
    'comet_ml>=3.31.12,<4.0.0',
]

extra_deps['tensorboard'] = [
    'tensorboard>=2.9.1,<3.0.0',
]

extra_deps['unet'] = [
    'monai>=0.9.1,<1.4',
    'scikit-learn>=1.0.1,<2',
]

extra_deps['vit'] = [
    'vit_pytorch==1.6.1',
]

extra_deps['timm'] = [
    'timm>=0.5.4,<0.6',
]

extra_deps['coco'] = [
    'pycocotools>=2.0.4,<3',
]

extra_deps['nlp'] = [
    'transformers>=4.11,<4.36,!=4.34.0',
    'datasets>=2.4,<3',
]

extra_deps['sentencepiece'] = [
    'protobuf<3.21',
    'sentencepiece==0.1.99',
]

extra_deps['mlperf'] = [
    # TODO: use pip when available: https://github.com/mlcommons/logging/issues/218
    # "mlperf_logging @ git+https://github.com/mlperf/logging.git",
    'py-cpuinfo>=8.0.0,<10',
]

extra_deps['streaming'] = [
    'mosaicml-streaming<1.0',
    'boto3>=1.21.45,<2',
    'paramiko>=2.11.0,<3',
]

extra_deps['libcloud'] = [
    'apache-libcloud>=3.3.1,<4',
]

extra_deps['oci'] = [
    'oci>=2.88.2,<3.0.0',
]

extra_deps['gcs'] = [
    'google-cloud-storage>=2.0.0,<3.0',
]

extra_deps['onnx'] = [
    'onnx>=1.12.0,<2',
    'onnxruntime>=1.12.1,<2',
]

extra_deps['mlflow'] = [
    'mlflow>=2.5.0,<3.0',
]

extra_deps['pandas'] = ['pandas>=2.0.0,<3.0']

extra_deps['databricks'] = ['databricks-sdk>=0.8.0,<1.0']

extra_deps['all'] = {dep for deps in extra_deps.values() for dep in deps}

composer_data_files = ['py.typed']
composer_data_files += package_files('composer', 'yamls', '.yaml')
composer_data_files += package_files('composer', 'algorithms', '.json')

package_name = os.environ.get('COMPOSER_PACKAGE_NAME', 'mosaicml')

if package_name != 'mosaicml':
    print(f'`Building composer as `{package_name}`)', file=sys.stderr)

setup(name=package_name,
      version=composer_version,
      author='MosaicML',
      author_email='team@mosaicml.com',
      description=('Composer is a PyTorch library that enables you to train ' +
                   'neural networks faster, at lower cost, and to higher accuracy.'),
      long_description=long_description,
      long_description_content_type='text/markdown',
      url='https://github.com/mosaicml/composer',
      include_package_data=True,
      package_data={
          'composer': composer_data_files,
      },
      packages=setuptools.find_packages(exclude=['docker*', 'examples*', 'scripts*', 'tests*']),
      classifiers=[
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.8',
          'Programming Language :: Python :: 3.9',
          'Programming Language :: Python :: 3.10',
      ],
      install_requires=install_requires,
      entry_points={
          'console_scripts': [
              'composer = composer.cli.launcher:main',
              'composer_collect_env = composer.utils.collect_env:main',
              'composer_validate_remote_path = composer.utils.file_helpers:validate_remote_path',
          ],
      },
      extras_require=extra_deps,
      dependency_links=['https://developer.download.nvidia.com/compute/redist'],
      python_requires='>=3.8',
      ext_package='composer',
      cmdclass={'develop': develop})

# only visible if user installs with verbose -v flag
# Printing to stdout as not to interfere with setup.py CLI flags (e.g. --version)
print('*' * 20, file=sys.stderr)
print(textwrap.dedent("""\
    NOTE: For best performance, we recommend installing Pillow-SIMD
    for accelerated image processing operations. To install:
    \t pip uninstall pillow && pip install pillow-simd"""),
      file=sys.stderr)
print('*' * 20, file=sys.stderr)
