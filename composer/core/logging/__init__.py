# Copyright 2021 MosaicML. All Rights Reserved.

"""Super Badass Docstring about composer.core.logging

Attributes:
    TLogData: Logging attribute that does stuff
    TLogDataValue: Another attribute that probably does stuff
"""

from composer.core.logging._base_backend import LoggerCallback as LoggerCallback
from composer.core.logging._logger import Logger as Logger
from composer.core.logging._logger import LogLevel as LogLevel
from composer.core.logging._logger import TLogData as TLogData
from composer.core.logging._logger import TLogDataValue as TLogDataValue
from composer.core.logging._logger import format_log_data_value as format_log_data_value

__all__ = ["LoggerCallback", "Logger", "LogLevel", "TLogData", "TLogDataValue", "format_log_data_value"]

LoggerCallback.__module__ = __name__
Logger.__module__ = __name__
LogLevel.__module__ = __name__
TLogData.__module__ = __name__
TLogDataValue.__module__ = __name__
format_log_data_value.__module__ = __name__