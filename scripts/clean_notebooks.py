# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Utility script to clean jupyter notebooks.

Removes any outputs, all cell metadata except for 'tags', and also remove the notebook metadata "widgets".

.. argparse::
   :filename: scripts/clean_notebooks.py
   :func: _get_parser
   :prog: clean_notebooks.py

"""
import argparse
import pathlib

import nbformat


def clean_notebook(notebook: nbformat.NotebookNode):
    """Clean jupyter notebooks.

    Removes any outputs, all cell metadata except for 'tags', and also remove the notebook metadata "widgets".

    Args:
        notebook (nbformat.NotebookNode): The notebook

    Returns:
        None: modifies the notebook in place.
    """
    for cell in notebook.cells:
        if cell["cell_type"] == 'code':
            cell["execution_count"] = None
            cell["outputs"] = []

        tags = cell["metadata"].get("tags", None)
        cell["metadata"] = {}

        if tags:
            cell["metadata"]["tags"] = tags

    if "widgets" in notebook["metadata"]:
        del notebook["metadata"]["widgets"]


def _get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", "-i", nargs="*", metavar="PATH", type=pathlib.Path)
    return parser


if __name__ == '__main__':
    parser = _get_parser()
    args = parser.parse_args()

    for input in args.inputs:
        notebook = nbformat.read(input, as_version=nbformat.NO_CONVERT)
        clean_notebook(notebook)
        nbformat.write(notebook, input)
