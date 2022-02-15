"""Utility script to clean jupyter notebooks.

Removes any outputs, all cell metadata except for 'tags', and also remove the notebook metadata "widgets".
"""

import argparse
import pathlib

import nbformat


def clean_notebook(notebook: nbformat.NotebookNode,) -> nbformat.NotebookNode:

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

    return notebook


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", "-i", nargs="*", metavar="PATH", type=pathlib.Path)
    args = parser.parse_args()

    for input in args.inputs:
        notebook = nbformat.read(input, as_version=nbformat.NO_CONVERT)
        notebook = clean_notebook(notebook)
        nbformat.write(notebook, input)
