from datetime import datetime as dt
import sys
import os


FATAL_ERROR = "FATAL"
WARNING = "WARNING"
INFORMATION = "INFO"
NON_FATAL_ERROR = "ERROR"
OUTPUT = "OUTPUT"


class Logger:

    def __init__(self):
        if 'LOGGING_SUPPRESSED_TYPES' in os.environ:
            self._suppressed_types = os.environ['LOGGING_SUPPRESSED_TYPES'].split(',')
        else:
            self._suppressed_types = []
    _BASE_MSG = "[{}] {}: {}"

    def log(self, msg_type: str, msg: str) -> None:
        if msg_type in self._suppressed_types:
            return
        time = dt.now()
        time_str = time.isoformat()
        time_str = time_str[11:19]
        msg_out = self._BASE_MSG.format(msg_type, time_str, msg)
        dest_stream = sys.stderr
        if msg_type in [INFORMATION, OUTPUT]:
            dest_stream = sys.stdout
        print(msg_out, file=dest_stream)


try:
    logger = logger
except NameError:
    logger = Logger()
