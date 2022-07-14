import concurrent.futures
import functools
import os
from typing import Any, Callable, Dict

import torch_xla.core.xla_model as xm


def pjrt_device() -> str:
  return os.environ['PJRT_DEVICE']


def num_visible_tpu_chips() -> int:
  visible_devices = os.environ.get('TPU_VISIBLE_DEVICES')

  return len(visible_devices.split(',')) if visible_devices else 4


def configure_tpu_topology(rank: int, processes: int, base_port=8476):
  '''Set default TPU topology environment variables for a single TPU host.'''
  ports = list(range(base_port, base_port + processes))
  os.environ.setdefault('TPU_CHIPS_PER_PROCESS_BOUNDS', '1,1,1')
  os.environ.setdefault('TPU_PROCESS_BOUNDS', '2,2,1')
  os.environ.setdefault('TPU_PROCESS_ADDRESSES',
                        ','.join(f'localhost:{port}' for port in ports))

  os.environ.setdefault('TPU_VISIBLE_DEVICES', str(rank))
  os.environ.setdefault('TPU_PROCESS_PORT', str(ports[rank]))
  os.environ.setdefault('CLOUD_TPU_TASK_ID', str(rank))


def run_thread_per_device(rank: int, processes: int,
                          fn: Callable) -> Dict[int, Any]:
  '''Run `fn` in a separate thread on each visible device.

  Args:
    rank: rank of current process
    processes: number of processes on this host
    fn: Function to run on all devices

  Returns:
    Dict of the form {thread_rank: return_value}, where return_value is the
    result of calling `fn`.
  '''
  if pjrt_device() == 'TPU':
    configure_tpu_topology(rank, processes)

  xm.set_replication(xm.xla_device(), xm.get_xla_supported_devices())
  threads = len(xm.get_xla_supported_devices())
  os.environ['XRT_SHARD_WORLD_SIZE'] = str(processes * threads)

  with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
    futures = {executor.submit(fn, index=i): i for i in range(threads)}

    results = {
        futures[f]: f.result() for f in concurrent.futures.as_completed(futures)
    }

  return results


def run_pjrt_multiprocess(fn: Callable, *args,
                          **kwargs) -> Dict[int, Dict[int, Any]]:
  '''Run `fn` on all devices available to PjRt.

  Args:
    fn: Function to run on all devices
    args: args to pass to `fn`
    kwargs: kwargs to pass to `fn`

  Returns:
    Dict of the form {process_rank: {thread_rank: return_value}}, where
    return_value is the result of calling `fn`.
  '''
  if pjrt_device() == 'TPU':
    processes = num_visible_tpu_chips()
  else:
    processes = 1

  with concurrent.futures.ProcessPoolExecutor(
      max_workers=processes) as executor:
    futures = {
        executor.submit(run_thread_per_device, i, processes,
                        functools.partial(fn, *args, **kwargs)): i
        for i in range(processes)
    }

    results = {
        futures[f]: f.result() for f in concurrent.futures.as_completed(futures)
    }

  return results
