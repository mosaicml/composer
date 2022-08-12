# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Merge trace files together.

To run:

.. code-block::

    python -m composer.profiler.json_trace_merger -o merged_trace_output.json path/to/input_file_1.json path/to/input_file_2.json ...

To view the traces, open a Google Chrome browser window, navigate to ``chrome://tracing`` and load the ``merged_trace_output.json``
to visualize the trace.
"""
import argparse
import json
import pathlib
from typing import Dict, List, Tuple, Union

__all__ = ['merge_traces']


def _load_trace(file: Union[str, pathlib.Path]) -> Union[Dict, List]:
    with open(file, 'r') as f:
        trace_str = f.read().strip()
        if trace_str.startswith('['):
            if trace_str.endswith('}'):
                trace_str += ']'
            if trace_str.endswith(','):
                trace_str = trace_str[-1] + ']'
        return json.loads(trace_str)


def _get_global_rank_from_file(file: Union[str, pathlib.Path]) -> int:
    trace_json = _load_trace(file)
    if isinstance(trace_json, list):
        for event in trace_json:
            if event['ph'] == 'M' and event['name'] == 'global_rank':
                return event['args']['value']
    else:
        assert isinstance(trace_json, dict)
        return trace_json['global_rank']
    raise RuntimeError('global rank not found in file')


def _get_rank_to_clock_syncs(trace_files: Tuple[Union[str, pathlib.Path], ...]) -> Dict[int, int]:
    rank_to_clock_sync: Dict[int, int] = {}
    for filename in trace_files:
        rank = _get_global_rank_from_file(filename)
        trace_json = _load_trace(filename)
        if isinstance(trace_json, list):
            for event in trace_json:
                if event['ph'] == 'M' and event['name'] == 'clock_sync_timestamp_us':
                    clock_sync = event['args']['value']
                    rank_to_clock_sync[rank] = clock_sync
                    break
        else:
            assert isinstance(trace_json, dict)
            if trace_json.get('clock_sync_timestamp_us') is not None:
                rank_to_clock_sync[rank] = trace_json['clock_sync_timestamp_us']

    return rank_to_clock_sync


def merge_traces(output_file: Union[str, pathlib.Path], *trace_files: Union[str, pathlib.Path]):
    """Merge profiler output JSON trace files together.

    This function will update the trace events such that:

    - The ``pid`` will be set to the global rank.
    - The ``ts`` is synchronized with that of the rank 0 process.
    - The backward pass process appears below the forward process.

    Args:
        output_file (str | pathlib.Path): The file to write the merged trace to
        trace_files (str | pathlib.Path): Variable number of trace files to merge together
    """
    ranks_to_clock_sync = _get_rank_to_clock_syncs(trace_files)
    rank_to_backwards_thread = {}
    rank_to_seen_threads = {rank: set() for rank in ranks_to_clock_sync.keys()}

    # Local rank zero will be the lowest global rank
    # Use that as the base timestamp for clock syncing
    lowest_rank = float('inf')
    for k in ranks_to_clock_sync:
        lowest_rank = min(k, lowest_rank)
    assert isinstance(lowest_rank, int), 'there should be at least one rank'
    rank_zero_clock_sync = ranks_to_clock_sync[lowest_rank]

    with open(output_file, 'w+') as output_f:
        is_first_line = True
        output_f.write('[')
        for trace_filename in trace_files:
            rank = _get_global_rank_from_file(trace_filename)
            clock_sync_diff = rank_zero_clock_sync - ranks_to_clock_sync[rank]

            with open(trace_filename, 'r') as trace_f:
                trace_data = json.load(trace_f)

                if isinstance(trace_data, list):
                    trace_list = trace_data
                else:
                    assert isinstance(trace_data, dict)
                    trace_list = trace_data['traceEvents']

                for event in trace_list:
                    if 'pid' not in event:
                        # we need the pid to merge
                        continue
                    if 'tid' not in event:
                        continue
                    if 'PyTorch Profiler' in str(event['tid']):
                        # skip this line; it pollutes the UI
                        continue
                    if 'ts' in event:
                        event['ts'] = event['ts'] + clock_sync_diff
                    event['pid'] = rank
                    if event['tid'] not in rank_to_seen_threads[rank]:
                        # By default, make all threads display last
                        # The training loop thread later sets itself as thread 0
                        # and the backwards pass thread is set as thread 1
                        if not is_first_line:
                            output_f.write(',')
                        output_f.write('\n    ')
                        json.dump(
                            {
                                'name': 'thread_sort_index',
                                'ph': 'M',
                                'pid': rank,
                                'tid': event['tid'],
                                'args': {
                                    'sort_index': 99999,
                                }
                            }, output_f)
                        rank_to_seen_threads[rank].add(event['tid'])
                        is_first_line = False
                    if event['name'] == 'MulBackward0':
                        rank_to_backwards_thread[rank] = event['tid']
                    if not is_first_line:
                        output_f.write(',')
                    is_first_line = False
                    output_f.write(f'\n    ')
                    json.dump(event, output_f)
        for pid, tid in rank_to_backwards_thread.items():
            output_f.write(',\n    ')
            json.dump({
                'name': 'thread_sort_index',
                'ph': 'M',
                'pid': pid,
                'tid': tid,
                'args': {
                    'sort_index': 1
                }
            }, output_f)

        output_f.write('\n]\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_files', type=str, nargs='+', help='Input files')
    parser.add_argument('-o', '--output_file', help='Output File', required=True)
    args = parser.parse_args()
    output_file = args.output_file
    input_files = args.input_files
    merge_traces(output_file, *input_files)
