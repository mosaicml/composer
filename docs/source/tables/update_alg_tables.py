# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Helper function to generate the README table."""
import json
import os
from pathlib import Path

import utils

import composer
from composer import functional as CF

EXCLUDE_METHODS = ['no_op_model', 'utils']

HEADER = ['Name', 'Functional', 'Attribution', 'tl;dr']
ATTRIBUTES = ['class_name', 'functional', 'tldr', 'attribution', 'link']
GITHUB_BASE = 'https://github.com/mosaicml/composer/tree/dev/composer/algorithms/'

folder_path = os.path.join(os.path.dirname(composer.__file__), 'algorithms')

methods = utils.list_dirs(Path(folder_path))
methods = [m for m in methods if m not in EXCLUDE_METHODS]

if not len(methods):
    raise ValueError(f'Found 0 methods in {folder_path}')

print(f'Found {len(methods)} methods with metadata.')

metadata = {}
for name in methods:
    json_path = os.path.join(folder_path, name, 'metadata.json')
    with open(json_path, 'r') as f:
        metadata[name] = json.load(f)[name]

        # test functional method is importable
        method_functional = metadata[name]['functional']
        if method_functional and not hasattr(CF, method_functional):
            raise ImportError(f'Unable to import functional form {method_functional} for {name}')

        metadata[name]['functional'] = f'`cf.{method_functional}`'
        metadata[name]['github_link'] = GITHUB_BASE + name

# define row format
row = [
    '[{class_name}]({github_link})',
    '{functional}',
    lambda d: '[{attribution}]({link})' if d['link'] else ['attribution'],
    '{tldr}',
]

table_md = utils.build_markdown_table(
    header=HEADER,
    metadata=metadata,
    sorted_keys=sorted(metadata.keys()),
    row_format=row,
)

table_path = os.path.join(os.path.dirname(__file__), 'algorithms_table.md')
with open(table_path, 'w') as f:
    f.write(table_md)

print(f'Table written to {table_path}')
