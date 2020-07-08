from datetime import datetime as dt
import sys


FATAL_ERROR = "FATAL"
WARNING = "WARNING"
INFORMATION = "INFO"
NON_FATAL_ERROR = "ERROR"


class Logger:

    _BASE_MSG = "[{}] {}: {}"

    def log(self, msg_type: str, msg: str) -> None:
        time = dt.now()
        time_str = time.isoformat()
        time_str = time_str[11:19]
        msg_out = self._BASE_MSG.format(msg_type, time_str, msg)
        dest_stream = sys.stderr
        if msg_type == INFORMATION:
            dest_stream = sys.stdout
        print(msg_out, file=dest_stream)


try:
    logger = logger
except NameError:
    logger = Logger()
