import json
import logging


class JsonLogFormatter(logging.Formatter):
    def __init__(self, dist, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dist = dist

    def format(self, record):
        log_record = {
            'asctime': self.formatTime(record, self.datefmt),
            'rank': f'rank{self.dist.get_global_rank()}',
            'process': record.process,
            'threadName': record.threadName,
            'levelname': record.levelname,
            'name': record.name,
            'message': record.getMessage(),
        }
        return json.dumps(log_record)
