import json
import logging
import traceback


class JsonLogFormatter(logging.Formatter):
    def __init__(self, dist=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dist = dist

    def format(self, record):
        log_record = {
            'asctime': self.formatTime(record, self.datefmt),
            'rank': f'rank{self.dist.get_global_rank()}' if self.dist else 'unknown',
            'process': record.process,
            'threadName': record.threadName,
            'levelname': record.levelname,
            'name': record.name,
            'message': record.getMessage(),
        }
        # if record.exc_info:
        #     log_record['exception'] = str(record.exc_info[1])
        #     log_record['traceback'] = traceback.format_exception(*record.exc_info)
        return json.dumps(log_record)
