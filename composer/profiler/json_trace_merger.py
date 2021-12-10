# Copyright 2021 MosaicML. All Rights Reserved.

"""Merge trace files together.

To run:

.. code-block:: console

    python -m composer.profiler.json_trace_merger -o merged_trace_output.json path/to/input_file_1.json path/to/input_file_2.json ...

Then, open the `Chrome Trace Viewer <chrome://tracing>`_ in a browser window, and upload
``merged_trace_output.json`` to visualize the trace.
"""
import argparse
import json
from typing import Dict, List, Tuple, Union


def _load_trace(file: str) -> Union[Dict, List]:
    with open(file, "r") as f:
        trace_str = f.read().strip()
        if trace_str.startswith("["):
            if trace_str.endswith("}"):
                trace_str += "]"
            if trace_str.endswith(","):
                trace_str = trace_str[-1] + "]"
        return json.loads(trace_str)


def _get_global_rank_from_file(file: str) -> int:
    trace_json = _load_trace(file)
    if isinstance(trace_json, list):
        for event in trace_json:
            if event["ph"] == "M" and event["name"] == "global_rank":
                return event["args"]["value"]
    else:
        assert isinstance(trace_json, dict)
        return trace_json["global_rank"]
    raise RuntimeError("global rank not found in file")


def _get_rank_to_clock_syncs(trace_files: Tuple[str, ...]) -> Dict[int, int]:
    rank_to_clock_sync: Dict[int, int] = {}
    for filename in trace_files:
        rank = _get_global_rank_from_file(filename)
        with open(filename, "r") as f:
            # TODO convert to valid JSON if the file isn't valid json
            trace_json = json.load(f)
            if isinstance(trace_json, list):
                for event in trace_json:
                    if event["ph"] == "M" and event["name"] == "clock_sync_timestamp_us":
                        clock_sync = event["args"]["value"]
                        rank_to_clock_sync[rank] = clock_sync
                        break
            else:
                assert isinstance(trace_json, dict)
                if trace_json.get("clock_sync_timestamp_us") is not None:
                    rank_to_clock_sync[rank] = trace_json["clock_sync_timestamp_us"]

    return rank_to_clock_sync


def merge_traces(output_file: str, *trace_files: str):
    """Merge trace files together.

    Compute the clock sync difference across every process and add the difference
    to each recorded trace event.

    Each trace file must contain ``global_rank`` in the metadata, and
    one trace file per rank must contain ``clock_sync_timestamp_us`` in the metadata.

    Args:
        output_file (str): The file to write the merged trace to
        trace_files (str): Variable number of trace files to merge together
    """

    ranks_to_clock_sync = _get_rank_to_clock_syncs(trace_files)

    rank_zero_clock_sync = ranks_to_clock_sync[0]

    output_buffer = []
    for trace_filename in trace_files:
        rank = _get_global_rank_from_file(trace_filename)
        clock_sync_diff = rank_zero_clock_sync - ranks_to_clock_sync[rank]

        with open(trace_filename, "r") as f:
            trace_data = json.load(f)

            if isinstance(trace_data, list):
                trace_list = trace_data
            else:
                assert isinstance(trace_data, dict)
                trace_list = trace_data["traceEvents"]

            for event in trace_list:
                if "pid" not in event:
                    # we need the pid to merge
                    continue
                if "ts" in event:
                    event["ts"] = event["ts"] + clock_sync_diff
                output_buffer.append(event)

    with open(output_file, "x") as f:
        json.dump(output_buffer, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input_files', type=str, nargs='+', help='Input files')
    parser.add_argument("-o", "--output_file", help="Output File", required=True)
    args = parser.parse_args()
    output_file = args.output_file
    input_files = args.input_files
    merge_traces(output_file, *input_files)
