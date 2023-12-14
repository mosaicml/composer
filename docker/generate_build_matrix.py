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
PRODUCTION_PYTORCH_VERSION = '2.1.1'


def _get_torchvision_version(pytorch_version: str):
    if pytorch_version == '2.1.1':
        return '0.16.1'
    if pytorch_version == '2.0.1':
        return '0.15.2'
    if pytorch_version == '1.13.1':
        return '0.14.1'
    raise ValueError(f'Invalid pytorch_version: {pytorch_version}')


def _get_base_image(cuda_version: str):
    if not cuda_version:
        return 'ubuntu:20.04'
    return f'nvidia/cuda:{cuda_version}-cudnn8-devel-ubuntu20.04'


def _get_cuda_version(pytorch_version: str, use_cuda: bool):
    if not use_cuda:
        return ''
    if pytorch_version == '2.1.1':
        return '12.1.0'
    if pytorch_version == '2.0.1':
        return '11.8.0'
    if pytorch_version == '1.13.1':
        return '11.7.1'
    raise ValueError(f'Invalid pytorch_version: {pytorch_version}')


def _get_cuda_version_tag(cuda_version: str):
    if not cuda_version:
        return 'cpu'
    return 'cu' + ''.join(cuda_version.split('.')[:2])


def _get_cuda_override(cuda_version: str):
    if cuda_version == '12.1.0':
        cuda_121_override_string = ('cuda>=12.1 brand=tesla,driver>=450,driver<451 '
                                    'brand=tesla,driver>=470,driver<471 brand=unknown,driver>=470,driver<471 '
                                    'brand=nvidia,driver>=470,driver<471 brand=nvidiartx,driver>=470,driver<471 '
                                    'brand=geforce,driver>=470,driver<471 brand=geforcertx,driver>=470,driver<471 '
                                    'brand=quadro,driver>=470,driver<471 brand=quadrortx,driver>=470,driver<471 '
                                    'brand=titan,driver>=470,driver<471 brand=titanrtx,driver>=470,driver<471 '
                                    'brand=tesla,driver>=510,driver<511 brand=unknown,driver>=510,driver<511 '
                                    'brand=nvidia,driver>=510,driver<511 brand=nvidiartx,driver>=510,driver<511 '
                                    'brand=geforce,driver>=510,driver<511 brand=geforcertx,driver>=510,driver<511 '
                                    'brand=quadro,driver>=510,driver<511 brand=quadrortx,driver>=510,driver<511 '
                                    'brand=titan,driver>=510,driver<511 brand=titanrtx,driver>=510,driver<511 '
                                    'brand=tesla,driver>=515,driver<516 brand=unknown,driver>=515,driver<516 '
                                    'brand=nvidia,driver>=515,driver<516 brand=nvidiartx,driver>=515,driver<516 '
                                    'brand=geforce,driver>=515,driver<516 brand=geforcertx,driver>=515,driver<516 '
                                    'brand=quadro,driver>=515,driver<516 brand=quadrortx,driver>=515,driver<516 '
                                    'brand=titan,driver>=515,driver<516 brand=titanrtx,driver>=515,driver<516 '
                                    'brand=tesla,driver>=525,driver<526 brand=unknown,driver>=525,driver<526 '
                                    'brand=nvidia,driver>=525,driver<526 brand=nvidiartx,driver>=525,driver<526 '
                                    'brand=geforce,driver>=525,driver<526 brand=geforcertx,driver>=525,driver<526 '
                                    'brand=quadro,driver>=525,driver<526 brand=quadrortx,driver>=525,driver<526 '
                                    'brand=titan,driver>=525,driver<526 brand=titanrtx,driver>=525,driver<526')

        return cuda_121_override_string

    if cuda_version == '11.8.0':
        cuda_118_override_string = ('cuda>=11.8 brand=tesla,driver>=470,driver<471 '
                                    'brand=tesla,driver>=515,driver<516 brand=unknown,driver>=470,driver<471 '
                                    'brand=unknown,driver>=515,driver<516 brand=nvidia,driver>=470,driver<471 '
                                    'brand=nvidia,driver>=515,driver<516 brand=nvidiartx,driver>=470,driver<471 '
                                    'brand=nvidiartx,driver>=515,driver<516 brand=geforce,driver>=470,driver<471 '
                                    'brand=geforce,driver>=515,driver<516 brand=quadro,driver>=470,driver<471 '
                                    'brand=quadro,driver>=515,driver<516 brand=titan,driver>=470,driver<471 '
                                    'brand=titan,driver>=515,driver<516 brand=titanrtx,driver>=470,driver<471 '
                                    'brand=titanrtx,driver>=515,driver<516')

        return cuda_118_override_string

    return ''


def _get_pytorch_tags(python_version: str, pytorch_version: str, cuda_version: str, stage: str, interconnect: str):
    if stage == 'pytorch_stage':
        base_image_name = 'mosaicml/pytorch'
    else:
        raise ValueError(f'Invalid stage: {stage}')
    cuda_version_tag = _get_cuda_version_tag(cuda_version)
    tags = [f'{base_image_name}:{pytorch_version}_{cuda_version_tag}-python{python_version}-ubuntu20.04']

    if python_version == LATEST_PYTHON_VERSION and pytorch_version == PRODUCTION_PYTORCH_VERSION:
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
    python_versions = ['3.10']
    pytorch_versions = ['2.1.1', '2.0.1', '1.13.1']
    cuda_options = [True, False]
    stages = ['pytorch_stage']
    interconnects = ['mellanox', 'EFA']  # mellanox is default, EFA needed for AWS

    pytorch_entries = []

    for product in itertools.product(python_versions, pytorch_versions, cuda_options, stages, interconnects):
        python_version, pytorch_version, use_cuda, stage, interconnect = product

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
            'TAGS':
                _get_pytorch_tags(
                    python_version=python_version,
                    pytorch_version=pytorch_version,
                    cuda_version=cuda_version,
                    stage=stage,
                    interconnect=interconnect,
                ),
            'PYTORCH_NIGHTLY_URL':
                '',
            'PYTORCH_NIGHTLY_VERSION':
                '',
            'NVIDIA_REQUIRE_CUDA_OVERRIDE':
                _get_cuda_override(cuda_version),
        }

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
            entry['AWS_OFI_NCCL_VERSION'] = 'v1.7.4-aws'

        pytorch_entries.append(entry)
    nightly_entry = {
        'AWS_OFI_NCCL_VERSION': '',
        'BASE_IMAGE': 'nvidia/cuda:12.1.0-cudnn8-devel-ubuntu20.04',
        'CUDA_VERSION': '12.1.0',
        'IMAGE_NAME': 'torch-nightly-2-2-0-20231024-cu121',
        'MOFED_VERSION': '5.5-1.0.3.2',
        'NVIDIA_REQUIRE_CUDA_OVERRIDE': _get_cuda_override('12.1.0'),
        'PYTHON_VERSION': '3.10',
        'PYTORCH_VERSION': '2.2.0',
        'PYTORCH_NIGHTLY_URL': 'https://download.pytorch.org/whl/nightly/cu121',
        'PYTORCH_NIGHTLY_VERSION': 'dev20231024+cu121',
        'TAGS': ['mosaicml/pytorch:2.2.0_cu121-nightly20231024-python3.10-ubuntu20.04'],
        'TARGET': 'pytorch_stage',
        'TORCHVISION_VERSION': '0.17.0'
    }
    pytorch_entries.append(nightly_entry)
    composer_entries = []

    # The `GIT_COMMIT` is a placeholder and Jenkins will substitute it with the actual git commit for the `composer_staging` images
    composer_versions = ['0.17.2']  # Only build images for the latest composer version
    composer_python_versions = [LATEST_PYTHON_VERSION]  # just build composer against the latest

    for product in itertools.product(composer_python_versions, composer_versions, cuda_options):
        python_version, composer_version, use_cuda = product

        pytorch_version = PRODUCTION_PYTORCH_VERSION
        cuda_version = _get_cuda_version(pytorch_version=pytorch_version, use_cuda=use_cuda)
        cpu = '-cpu' if not use_cuda else ''

        entry = {
            'IMAGE_NAME': f"composer-{composer_version.replace('.', '-')}{cpu}",
            'BASE_IMAGE': _get_base_image(cuda_version),
            'CUDA_VERSION': cuda_version,
            'PYTHON_VERSION': python_version,
            'PYTORCH_VERSION': pytorch_version,
            'PYTORCH_NIGHTLY_URL': '',
            'PYTORCH_NIGHTLY_VERSION': '',
            'TARGET': 'composer_stage',
            'TORCHVISION_VERSION': _get_torchvision_version(pytorch_version),
            'MOFED_VERSION': '5.5-1.0.3.2',
            'AWS_OFI_NCCL_VERSION': '',
            'COMPOSER_INSTALL_COMMAND': f'mosaicml[all]=={composer_version}',
            'TAGS': _get_composer_tags(
                composer_version=composer_version,
                use_cuda=use_cuda,
            ),
            'NVIDIA_REQUIRE_CUDA_OVERRIDE': _get_cuda_override(cuda_version),
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
            'Base',  # Flavor
            entry['PYTORCH_VERSION'],  # Pytorch version
            cuda_version,  # Cuda version
            entry['PYTHON_VERSION'],  # Python version,
            ', '.join(reversed([f'`{x}`' for x in entry['TAGS']])),  # Docker tags
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
            ', '.join(reversed([f'`{x}`' for x in entry['TAGS']])),  # Docker tags
        ])
    table.sort(key=lambda x: x[1], reverse=True)  # cuda support
    table.sort(key=lambda x: packaging.version.parse('9999999999999'
                                                     if x[0] == 'latest' else x[0]), reverse=True)  # Composer version
    table_contents = tabulate.tabulate(table, headers, tablefmt='github', floatfmt='', disable_numparse=True)
    _write_table('COMPOSER_BUILD_MATRIX', table_contents)

    print('Successfully updated `build_matrix.yaml` and `README.md`.', file=sys.stderr)


if __name__ == '__main__':
    _main()
