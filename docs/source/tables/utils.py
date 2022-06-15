# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Helper functions for auto-generating tables from metadata."""
import importlib
import os
import shutil
import tempfile


def list_dirs(folder):
    """Lists all dirs for a given folder.

    Args:
        folder (str): The folder to list dirs for.
    """
    return sorted(
        child.name for child in folder.iterdir() if child.is_dir() and ('__pycache__' not in str(child.absolute())))


def assert_attributes_exist(name, module_dict, attributes):
    """Assert that module has the provided attributes.

    Args:
        name (str): The class name.
        module_dict (dict): The dict form of the class.
        attributes (list): The attributes to check for.
    """
    for attribute in attributes:
        assert attribute in module_dict, \
        f'{name} should define {attribute} in its __init__.py file.'


def get_metadata(names, attributes, module_basepath):
    """Returns a nested dict of metadata with names as keys.

    Checks that all attributes exist in module given by module_basepath.name.

    Args:
        names (str): The module names.
        attributes (list): The attributes to check for.
        module_basepath (str): The import path of the module.

    Example::
        >>> get_metadata(
                names=['blurpool', 'label_smoothing'],
                attributes=['_name', '_tldr'],
                module_basepath='composer.algorithms'
            )
        {'blurpool': {'_name': ..., '_tldr': ...},
         'label_smoothing': {'_name': ..., '_tldr': ...}}
    """
    metadata = {}

    for name in names:
        module = importlib.import_module(f'{module_basepath}.{name}')

        if hasattr(module, '_metadata'):
            for subname in getattr(module, '_metadata'):
                submodule_dict = getattr(module, '_metadata')[subname]
                assert_attributes_exist(f'{name}/{subname}', submodule_dict, attributes)

                metadata[subname] = {a: submodule_dict[a] for a in attributes}

                # check for attributes with empty strings
                for attribute in attributes:
                    if not metadata[subname][attribute]:
                        print(f'WARNING: {subname} has empty metadata {attribute}')
        else:
            module_dict = module.__dict__

            assert_attributes_exist(name, module_dict, attributes)

            metadata[name] = {a: module_dict[a] for a in attributes}

            # check for attributes with empty strings
            for attribute in attributes:
                if not metadata[name][attribute]:
                    print(f'WARNING: {name} has empty metadata {attribute}')
    return metadata


def build_markdown_table(header, metadata, sorted_keys, row_format):
    """Builds a markdown table, formatting `row_format` with the `metadata`.

    Entries in the table are ordered by `sorted_keys`.

    Args:
        header (list): list of header strings
        metadata (dict): nested dict of metadata
        sorted_keys (list): order of rows in table
        row_format (list): list of length(header). Elements are either a string
            or a single-argument callable that returns a string.

    Returns:
        table_md (list): table in markdown format
    """
    table_md = _print_row(header)
    table_md += _print_row(['-' * len(h) for h in header])

    for name in sorted_keys:
        d = metadata[name]

        # single-argument callable that returns a string is used for conditional formats
        # e.g. to only print link if provided, define
        #    lambda d: '[Link]({_link})' if d[_link] else ''
        row = [r(d).format(**d) if callable(r) else r.format(**d) for r in row_format]
        table_md += _print_row(row)

    return table_md


def _print_row(row):
    return '|'.join(row) + '\n'


def index_tag_in_lines(lines, tag):
    """Returns line number where tag is found.

    Args:
        lines (list): List of lines to check.
        tag (str): Tag to find.
    """
    for index, line in enumerate(lines):
        if tag in line:
            return index
    raise ValueError(f'{tag} not found.')


def update_table_in_file(table, source_file):
    """Updates the table content based on a source file.

    Given a `source file`, updates the table. Searches
    the file for 'Table Start' and 'Table End' tags, and replaces
    the content between those tags.
    The original file is retained with the `.bkp` suffix.

    Args:
        table (list): list of strings
        source_file (path): path to source file
    """
    with open(source_file, 'r') as source, \
      tempfile.NamedTemporaryFile('w', delete=False) as temp:
        source_lines = source.readlines()

        table_start = index_tag_in_lines(source_lines, tag='Table Start')
        table_end = index_tag_in_lines(source_lines, tag='Table End')
        print(f'Found table_start tag at line no: {table_start}')
        print(f'Found table_end tag at line no: {table_end}')
        assert table_end > table_start, 'Table End must be after Table Start'

        table_written = False
        for line_no, line in enumerate(source_lines):
            if line_no <= table_start or line_no >= table_end:
                temp.write(line)
            elif not table_written:  # write table once
                temp.writelines(table)
                table_written = True

    backup_file = source_file.with_suffix('.md.bkp')
    os.rename(source_file, backup_file)
    print(f'Original file backed up at: {backup_file}')

    shutil.copy(temp.name, source_file)
