import json
import logging
import traceback


class JsonLogFormatter(logging.Formatter):

    def __init__(self, rank=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rank = rank

    def format(self, record):
        log_record = {
            'asctime': self.formatTime(record, self.datefmt),
            'rank': f'rank{self.rank}' if self.rank else 'unknown',
            'process': record.process,
            'threadName': record.threadName,
            'levelname': record.levelname,
            'name': record.name,
            'message': record.getMessage(),
        }
        if record.exc_info:
            log_record['exception'] = str(record.exc_info[1])
            log_record['traceback'] = traceback.format_exception(*record.exc_info)
        return json.dumps(log_record)
