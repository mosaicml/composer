# Copyright 2021 MosaicML. All Rights Reserved.

from __future__ import annotations

from copy import deepcopy
import logging
import os
from typing import Optional
# import aiohttp
# import asyncio

import time
from threading import Thread
from queue import Queue

from composer.core.logging import LogLevel, RankZeroLoggerBackend, TLogData
from composer.core.serializable import Serializable
from composer.core.types import Logger, State, StateDict

_MOSAIC_API_KEY_ENV = "MOSAIC_API_KEY"
_MOSAIC_LOGGER_URL = "https://api.mosaicml.com/v0/log/metric"
_JOB_ID_ENV = "MOSAIC_LOGGER_JOB_ID"
_SWEEP_ID_ENV = "MOSAIC_LOGGER_SWEEP_ID"

log = logging.getLogger(__name__)


class MosaicLoggerBackend(RankZeroLoggerBackend, Serializable):
    """Log to the MosaicML backend.

    Args:
        job_id (str, optional): The id of the job to write logs for.
        sweep_id (str, optional): The id of the sweep to write logs for.
        creds_file (str, optional): A file containing the MosaicML api_key. If not provided
            will default to the environment variable MOSAIC_API_KEY.
        flush_every_n_batches (int): Flush the log data buffer every n batches. (default: ``100``)
        max_logs_in_buffer (int): The maximum number of log entries allowed in the buffer before a
            forced flush. (default: ``1000``)
    """

    def __init__(self,
                 job_id: Optional[str] = None,
                 sweep_id: Optional[str] = None,
                 creds_file: Optional[str] = None,
                 flush_every_n_batches: int = 100,
                 max_logs_in_buffer: int = 1000) -> None:

        super().__init__()
        self.skip_logging = False
        self.job_id = job_id if job_id is not None else os.environ.get(_JOB_ID_ENV, None)
        if self.job_id is None:
            self.skip_logging = True
            log.warn("No job_id provided to MosaicLoggerBackend. This logger will be a no-op.")

        self.sweep_id = sweep_id if sweep_id is not None else os.environ.get(_SWEEP_ID_ENV, None)
        if self.sweep_id is None:
            self.skip_logging = True
            log.warn("No sweep_id provided to MosaicLoggerBackend. This logger will be a no-op.")

        if creds_file:
            with open(creds_file, 'r') as f:
                os.environ[_MOSAIC_API_KEY_ENV] = str(f.read())

        if os.environ.get(_MOSAIC_API_KEY_ENV, None) is None:
            self.skip_logging = True
            log.warn(f"No api_key set for environment variable {_MOSAIC_API_KEY_ENV}. This logger will be a no-op.")

        self.buffered_data = []
        # self.queue = asyncio.Queue()
        self.flush_every_n_batches = flush_every_n_batches
        self.max_logs_in_buffer = max_logs_in_buffer

        # self.loop = asyncio.new_event_loop()
        # self.loop.run_forever()

        self.queue = Queue()
        self.thread = Thread(target=self._listen_to_queue)
        self.thread.start() # do on training start


    def _log_metric(self, epoch: int, step: int, log_level: LogLevel, data: TLogData):
        del log_level  # unused

        if self.skip_logging:
            return

        log_data = {
            "job_id": self.job_id,
            "epoch": epoch,
            "step": step,
            "data": data,
        }
        self.buffered_data.append(log_data)
        if len(self.buffered_data) > self.max_logs_in_buffer:
            self._flush_buffered_data()

    # def batch_end(self, state: State, logger: Logger):
    #     del logger  # unused
    #     if (state.step + 1) % self.flush_every_n_batches == 0:
    #         self._flush_buffered_data()

    def training_end(self, state: State, logger: Logger):
        del state, logger  # unused

        # Flush any remaining logs on training end
        self._flush_buffered_data()

        # print('TRAINING END LOOP', self.loop)
        print('QUEUE SIZE:', self.queue.qsize())
        print('QUEUE:', self.queue)
        # self.queue.join()
        print('ALL QUEUE TASKS DONE')

        self.queue.put_nowait('STOP')
        self.thread.join()
        print('STOPPED THREAD - EXITING')
        # Block on all log writes finishing
        # self.loop.run_until_complete(self.queue.join())
        # self.loop.create_task(self.queue.join())
        # self.loop.run_in_executor


    def state_dict(self) -> StateDict:
        # Storing these fields in the state dict to support run resuming in the future.
        return {"job_id": self.job_id, "sweep_id": self.sweep_id}

    def load_state_dict(self, state: StateDict) -> None:
        self.job_id = state["job_id"]
        self.sweep_id = state["sweep_id"]

    def _listen_to_queue(self):
        print('THREAD STARTED RUNNING')
        while True:
            print('AT TOP OF LOOP BEFORE GET')
            data = self.queue.get(block=True)
            print('GOT DATA')
            if data == 'STOP':
                self.queue.task_done()
                print('stopping thread')
                return
            self._send_data(data)
            print('ABOUT TO CALL TASK DONE')
            self.queue.task_done()
            print('CALLED TASK DONE')


    def _flush_buffered_data(self):
        if len(self.buffered_data) == 0:
            return

        data_to_write = deepcopy(self.buffered_data.copy())
        self.buffered_data = []

        self.queue.put_nowait(data_to_write)

        # loop = asyncio.get_event_loop()
        # loop.
        # print('loop is', asyncio.get_event_loop())
        # print('new loop', asyncio.new_event_loop())
        # asyncio.create_task
        # Create a separate task for the new data written to the queue
        # self.loop.create_task(self._send_data(), name="ello")
        # self.loop.run_until_complete(self._send_data())
        # asyncio.create_task(self._send_data())
        # asyncio.get_event_loop().create_task(self._send_data())


    def _send_data(self, data):
        print(f'CALLED SEND DATA WITH DATA:', data)
        time.sleep(1)


    # async def _send_data(self) -> str:
    #     print('CALLED SEND DATA!!!')
    #     data = await self.queue.get()
    #     try:
    #         print('hi')
    #         # async with aiohttp.ClientSession() as session:
    #         #     async with session.post(_MOSAIC_LOGGER_URL,
    #         #                             headers={"X-MosaicML-API-key": os.environ.get(_MOSAIC_API_KEY_ENV, "")},
    #         #                             json={
    #         #                                 "experimentID": self.job_id,
    #         #                                 "runID": self.sweep_id,
    #         #                                 "data": data
    #         #                             }) as resp:
    #         #         response = await resp.text()
    #         #         #self.queue.task_done()
    #         #         return response
    #     except Exception as e:
    #         log.error(f"MosaicLogger got exception {e} when writing logs.")
    #         # Mark the task done even if there is an exception so that the training loop does not get stuck
    #         #self.queue.task_done()
    #     finally:
    #         self.queue.task_done()



# if __name__ == '__main__':
#     import time
#     async def worker():
#         print('worker started')
#         time.sleep(10)
#         print('worker finished')

#     async def worker_caller(i):
#         print(f'worker caller {i} started')
#         await worker()
#         print(f'worker caller {i} finished')

#     loop = asyncio.new_event_loop()

#     tasks = []
#     for i in range(3):
#         t = loop.create_task(worker_caller(i))
#         tasks.append(t)
#     await asyncio.gather(*tasks, loop=loop)
    # worker_caller()


