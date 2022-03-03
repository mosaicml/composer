import glob
import os
import textwrap

import pytest
from testbook import testbook

import composer

NB_PATH = '../notebooks/up_and_running_with_composer.ipynb'
nb_root = os.path.join(os.path.dirname(composer.__file__), '..', 'notebooks')

NOTEBOOKS = [
    os.path.join(nb_root, nb) \
    for nb in glob.glob(os.path.join(nb_root, '*.ipynb')) \
]


def seek(tb, tag):
    """generator that seeks next cell with the metadata tag, and returns the cell index."""
    for idx, cell in enumerate(tb.cells):
        metadata = cell['metadata']
        if 'tags' in metadata and tag in metadata['tags']:
            yield idx


@pytest.fixture(params=NOTEBOOKS)
def tb(request):
    notebook_path = request.param
    with testbook(notebook_path) as tb:
        yield tb


def _assert_tags_correct(tb):
    """Checks that the trainer_fit tag exists whenever trainer.fit is called."""
    for cell in tb.cells:
        if cell['cell_type'] == 'code' and 'trainer.fit' in cell['source']:
            metadata = cell['metadata']
            if 'tags' not in metadata or 'trainer_fit' not in metadata['tags']:
                raise ValueError("In cell that starts with '{}', trainer.fit was found "
                                 "but the trainer_fit tag was not in metadata.".format(cell['source'][:40]))


@pytest.mark.timeout(120)
@pytest.mark.notebooks
def test_notebook(tb):
    _assert_tags_correct(tb)

    start = 0
    for stop in seek(tb, tag="trainer_fit"):
        # run until right before the next trainer.fit call
        tb.execute_cell(range(start, stop))

        try:
            # patch the trainer to only train for 5 batches
            tb.inject("""
                from composer.core import Time
                trainer.state.max_duration = Time.from_timestring('2ep')
                trainer.state.train_subset_num_batches = 2
            """)
            trainer = tb.ref("trainer")
            assert trainer.state.train_subset_num_batches == 2
        except Exception as e:
            raise Exception(
                textwrap.dedent("""
                Test failed to patch the trainer's max_duration and
                train_subset_num_batches before notebook cell {}.
                The 'trainer' variable and the 'trainer_fit' metadata
                tag must be attached to the cell calling trainer.fit().

                For testing purposes, the definition of the trainer
                variable must have occured before the trainer.fit cell
                so the test can patch these variables for a shorter
                runtime.
            """.format(stop))) from e
        start = stop
