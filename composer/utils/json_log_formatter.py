import json
import logging


class JsonLogFormatter(logging.Formatter):
    def format(self, record):
        log_record = {
            "status": record.levelname,
            "message": record.getMessage()
        }
        return json.dumps(log_record)
