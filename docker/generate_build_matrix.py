# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Helper script to generate the ``build_matrix.yaml`` and update ``README.md``.

Note: this script requires tabulate. Run ``pip install tabulate`` if not installed

To run::

    python generate_build_matrix.py
"""

import itertools
import os
import sys

import packaging.version
import tabulate
import yaml

LATEST_PYTHON_VERSION = '3.10'


def _get_pytorch_version(python_version: str):
    if python_version == '3.10':
        return '1.13.1'
    if python_version == '3.9':
        return '1.12.1'
    if python_version in '3.8':
        return '1.11.0'
    raise ValueError(f'Invalid python version: {python_version}')


def _get_torchvision_version(pytorch_version: str):
    if pytorch_version == '1.13.1':
        return '0.14.1'
    if pytorch_version == '1.12.1':
        return '0.13.1'
    if pytorch_version == '1.11.0':
        return '0.12.0'
    raise ValueError(f'Invalid pytorch_version: {pytorch_version}')


def _get_torchtext_version(pytorch_version: str):
    if pytorch_version == '1.13.1':
        return '0.14.1'
    if pytorch_version == '1.12.1':
        return '0.13.1'
    if pytorch_version == '1.11.0':
        return '0.12.0'

    raise ValueError(f'Invalid pytorch_version: {pytorch_version}')


def _get_base_image(cuda_version: str):
    if not cuda_version:
        return 'ubuntu:20.04'
    return f'nvidia/cuda:{cuda_version}-cudnn8-devel-ubuntu20.04'


def _get_cuda_version(pytorch_version: str, use_cuda: bool):
    if not use_cuda:
        return ''
    if pytorch_version == '1.13.1':
        return '11.7.1'
    if pytorch_version == '1.12.1':
        return '11.6.2'
    if pytorch_version == '1.11.0':
        return '11.5.2'
    raise ValueError(f'Invalid pytorch_version: {pytorch_version}')


def _get_cuda_version_tag(cuda_version: str):
    if not cuda_version:
        return 'cpu'
    return 'cu' + ''.join(cuda_version.split('.')[:2])


def _get_pytorch_tags(python_version: str, pytorch_version: str, cuda_version: str, stage: str, interconnect: str):
    if stage == 'pytorch_stage':
        base_image_name = 'mosaicml/pytorch'
    elif stage == 'vision_stage':
        base_image_name = 'mosaicml/pytorch_vision'
    else:
        raise ValueError(f'Invalid stage: {stage}')
    cuda_version_tag = _get_cuda_version_tag(cuda_version)
    tags = [f'{base_image_name}:{pytorch_version}_{cuda_version_tag}-python{python_version}-ubuntu20.04']

    if python_version == LATEST_PYTHON_VERSION:
        if not cuda_version:
            tags.append(f'{base_image_name}:latest_cpu')
        else:
            tags.append(f'{base_image_name}:latest')

    if interconnect == 'EFA':
        tags = [f'{tag}-aws' for tag in tags]

    return tags


def _get_composer_tags(composer_version: str, use_cuda: bool):
    base_image_name = 'mosaicml/composer'

    tags = []
    if not use_cuda:
        tags.append(f'{base_image_name}:{composer_version}_cpu')
        tags.append(f'{base_image_name}:latest_cpu')
    else:
        tags.append(f'{base_image_name}:{composer_version}')
        tags.append(f'{base_image_name}:latest')

    return tags


def _get_image_name(pytorch_version: str, cuda_version: str, stage: str, interconnect: str):
    pytorch_version = pytorch_version.replace('.', '-')
    cuda_version = _get_cuda_version_tag(cuda_version)

    if stage == 'pytorch_stage':
        stage = ''
    elif stage == 'vision_stage':
        stage = '-vision'
    else:
        raise ValueError(f'Invalid stage: {stage}')

    if interconnect == 'EFA':
        fabric = '-aws'
    else:
        fabric = ''

    return f'torch{stage}-{pytorch_version}-{cuda_version}{fabric}'


def _write_table(table_tag: str, table_contents: str):
    with open(os.path.join(os.path.dirname(__name__), 'README.md'), 'r') as f:
        contents = f.read()

    begin_table_tag = f'<!-- BEGIN_{table_tag} -->'
    end_table_tag = f'<!-- END_{table_tag} -->'

    pre = contents.split(begin_table_tag)[0]
    post = contents.split(end_table_tag)[1]
    new_readme = f'{pre}{begin_table_tag}\n{table_contents}\n{end_table_tag}{post}'

    with open(os.path.join(os.path.dirname(__name__), 'README.md'), 'w') as f:
        f.write(new_readme)


def _main():
    python_versions = ['3.8', '3.9', '3.10']
    cuda_options = [True, False]
    stages = ['pytorch_stage', 'vision_stage']
    interconnects = ['mellanox', 'EFA']  # mellanox is default, EFA needed for AWS

    pytorch_entries = []

    for product in itertools.product(python_versions, cuda_options, stages, interconnects):
        python_version, use_cuda, stage, interconnect = product

        pytorch_version = _get_pytorch_version(python_version)
        cuda_version = _get_cuda_version(pytorch_version=pytorch_version, use_cuda=use_cuda)

        entry = {
            'IMAGE_NAME':
                _get_image_name(pytorch_version, cuda_version, stage, interconnect),
            'BASE_IMAGE':
                _get_base_image(cuda_version),
            'CUDA_VERSION':
                cuda_version,
            'PYTHON_VERSION':
                python_version,
            'PYTORCH_VERSION':
                pytorch_version,
            'TARGET':
                stage,
            'TORCHVISION_VERSION':
                _get_torchvision_version(pytorch_version),
            'TORCHTEXT_VERSION':
                _get_torchtext_version(pytorch_version),
            'TAGS':
                _get_pytorch_tags(
                    python_version=python_version,
                    pytorch_version=pytorch_version,
                    cuda_version=cuda_version,
                    stage=stage,
                    interconnect=interconnect,
                ),
        }

        # Only build the vision image on latest python
        if stage == 'vision_stage' and python_version != LATEST_PYTHON_VERSION:
            continue

        # Only build EFA image on latest python with cuda on pytorch_stage
        if interconnect == 'EFA' and not (python_version == LATEST_PYTHON_VERSION and use_cuda and
                                          stage == 'pytorch_stage'):
            continue

        # Skip the mellanox drivers if not in the cuda images or using EFA
        if not cuda_version or interconnect == 'EFA':
            entry['MOFED_VERSION'] = ''
        else:
            entry['MOFED_VERSION'] = '5.5-1.0.3.2'

        # Skip EFA drivers if not using EFA
        if interconnect != 'EFA':
            entry['AWS_OFI_NCCL_VERSION'] = ''
        else:
            entry['AWS_OFI_NCCL_VERSION'] = 'v1.5.0-aws'

        pytorch_entries.append(entry)

    composer_entries = []

    # The `GIT_COMMIT` is a placeholder and Jenkins will substitute it with the actual git commit for the `composer_staging` images
    composer_versions = ['0.12.1']  # Only build images for the latest composer version
    composer_python_versions = [LATEST_PYTHON_VERSION]  # just build composer against the latest

    for product in itertools.product(composer_python_versions, composer_versions, cuda_options):
        python_version, composer_version, use_cuda = product

        pytorch_version = _get_pytorch_version(python_version)
        cuda_version = _get_cuda_version(pytorch_version=pytorch_version, use_cuda=use_cuda)
        cpu = '-cpu' if not use_cuda else ''

        entry = {
            'IMAGE_NAME': f"composer-{composer_version.replace('.', '-')}{cpu}",
            'BASE_IMAGE': _get_base_image(cuda_version),
            'CUDA_VERSION': cuda_version,
            'PYTHON_VERSION': python_version,
            'PYTORCH_VERSION': pytorch_version,
            'TARGET': 'composer_stage',
            'TORCHVISION_VERSION': _get_torchvision_version(pytorch_version),
            'TORCHTEXT_VERSION': _get_torchtext_version(pytorch_version),
            'MOFED_VERSION': '5.5-1.0.3.2',
            'AWS_OFI_NCCL_VERSION': '',
            'COMPOSER_INSTALL_COMMAND': f'mosaicml[all]=={composer_version}',
            'TAGS': _get_composer_tags(
                composer_version=composer_version,
                use_cuda=use_cuda,
            ),
        }

        composer_entries.append(entry)

    with open(os.path.join(os.path.dirname(__name__), 'build_matrix.yaml'), 'w+') as f:
        f.write('# This file is automatically generated by generate_build_matrix.py. DO NOT EDIT!\n')
        yaml.safe_dump(pytorch_entries + composer_entries, f)

    # Also print the table for the readme

    # PyTorch Table
    headers = ['Linux Distro', 'Flavor', 'PyTorch Version', 'CUDA Version', 'Python Version', 'Docker Tags']
    table = []
    for entry in pytorch_entries:
        interconnect = 'N/A'
        if entry['CUDA_VERSION']:
            if entry['MOFED_VERSION'] != '':
                interconnect = 'Infiniband'
            else:
                interconnect = 'EFA'
        cuda_version = f"{entry['CUDA_VERSION']} ({interconnect})" if entry['CUDA_VERSION'] else 'cpu'
        table.append([
            'Ubuntu 20.04',  # Linux distro
            'Base' if entry['TARGET'] == 'pytorch_stage' else 'Vision',  # Flavor
            entry['PYTORCH_VERSION'],  # Pytorch version
            cuda_version,  # Cuda version
            entry['PYTHON_VERSION'],  # Python version,
            ', '.join(reversed(list(f'`{x}`' for x in entry['TAGS']))),  # Docker tags
        ])
    table.sort(
        key=lambda x: x[3].replace('Infiniband', '1').replace('EFA', '2'))  # cuda version, put infiniband ahead of EFA
    table.sort(key=lambda x: packaging.version.parse(x[4]), reverse=True)  # python version
    table.sort(key=lambda x: packaging.version.parse(x[2]), reverse=True)  # pytorch version
    table.sort(key=lambda x: x[1])  # flavor
    table_contents = tabulate.tabulate(table, headers, tablefmt='github', floatfmt='', disable_numparse=True)
    _write_table('PYTORCH_BUILD_MATRIX', table_contents)

    # Composer Table
    # Also print the table for the readme
    headers = ['Composer Version', 'CUDA Support', 'Docker Tag']
    table = []
    for entry in composer_entries:

        if len(entry['TAGS']) == 0:
            continue

        table.append([
            entry['TAGS'][0].split(':')[1].replace('_cpu', ''),  # Composer version, or 'latest'
            'No' if entry['BASE_IMAGE'].startswith('ubuntu:') else 'Yes',  # Whether there is Cuda support
            ', '.join(reversed(list(f'`{x}`' for x in entry['TAGS']))),  # Docker tags
        ])
    table.sort(key=lambda x: x[1], reverse=True)  # cuda support
    table.sort(key=lambda x: packaging.version.parse('9999999999999'
                                                     if x[0] == 'latest' else x[0]), reverse=True)  # Composer version
    table_contents = tabulate.tabulate(table, headers, tablefmt='github', floatfmt='', disable_numparse=True)
    _write_table('COMPOSER_BUILD_MATRIX', table_contents)

    print('Successfully updated `build_matrix.yaml` and `README.md`.', file=sys.stderr)


if __name__ == '__main__':
    _main()
