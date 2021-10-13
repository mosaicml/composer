# Copyright 2021 MosaicML. All Rights Reserved.

"""
Helper function to generate the README table

"""
from pathlib import Path

from composer import functional as CF
from composer import utils

HEADER = ['Name', 'Functional', 'Attribution', 'tl;dr', 'Card']
ATTRIBUTES = ['_class_name', '_functional', '_tldr', '_attribution', '_link', '_method_card']
GITHUB_BASE = 'https://github.com/mosaicml/composer/tree/dev/composer/algorithms/'

folder_path = Path(__file__).parent
methods = utils.list_dirs(folder_path)

if not len(methods):
    raise ValueError(f'Found 0 methods in {folder_path}')

print(f'Found {len(methods)} methods with metadata.')

metadata = utils.get_metadata(
    names=methods,
    attributes=ATTRIBUTES,
    module_basepath='composer.algorithms',
)

# add extra keys
for name, md in metadata.items():

    # add github link attribute
    md['_github_link'] = GITHUB_BASE + name

    # test that functional form is importable
    if md['_functional']:
        method_functional = md['_functional']

        if not hasattr(CF, method_functional):
            raise ImportError(f'Unable to import functional form {method_functional}')

        md['_functional'] = '`CF.{}`'.format(md['_functional'])

# define row format
row = [
    '[{_class_name}]({_github_link})',
    '{_functional}',
    lambda d: '[{_attribution}]({_link})' if d['_link'] else d['_attribution'],
    '{_tldr}',
    lambda d: '[Link]({_method_card})' if d['_method_card'] else '',
]

table_md = utils.build_markdown_table(
    header=HEADER,
    metadata=metadata,
    sorted_keys=sorted(metadata.keys()),
    row_format=row,
)

# update table in README.md
source_file = Path(__file__).parent.joinpath('README.md')
utils.update_table_in_file(table_md, source_file)
