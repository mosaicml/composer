# Copyright 2021 MosaicML. All Rights Reserved.

import setuptools
from setuptools import setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

install_requires = [
    "pyyaml>=5.4.1",
    "tqdm>=4.62.3",
    "torchmetrics",
    "torch_optimizer",
    "torchvision>=0.9.0",
    "torch>=1.8.1",
    "argparse>=1.4.0",
    "yahp>=0.0.10",
]
extra_deps = {}

extra_deps['base'] = []

extra_deps['dev'] = [
    'junitparser>=2.1.1',
    'coverage[toml]>=5.5',
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
    'transformers',
    'datasets',
]

extra_deps['unet'] = [
    'medpy==0.4.0',
    'monai',
    'scikit-learn',
    'torch_optimizer>=0.1.0',
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
        "": ["*.yaml", "*.yml"],
    },
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
    install_requires=install_requires,
    entry_points={
        'console_scripts': ['composer = examples.run_mosaic_trainer:main',],
    },
    extras_require=extra_deps,
    dependency_links=['https://developer.download.nvidia.com/compute/redist'],
    python_requires='>=3.8',
    ext_package="composer",
)

# only visible if user installs with verbose -v flag
print("*" * 20)
print("\nNOTE: For best performance, we recommend installing Pillow-SIMD "
      "\nfor accelerated image processing operations. To install:"
      "\n\n\t pip uninstall pillow && pip install pillow-simd\n")
print("*" * 20)
