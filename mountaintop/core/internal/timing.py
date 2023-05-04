import datetime
import time
import os
import inspect


from mountaintop.runx.logx import loggerx


class Timer(object):
    def __init__(self, format_str='%d:%02d:%.2f'):
        """
        Args:
            format_str: format of hour-minute-second
        """
        self.format_str = format_str
        self._start = time.time()
        self._last = self._start

    def reset(self):
        """
        Reset timer.
        """
        self._start = time.time()
        self._last = self._start

    def tick(self):
        '''
        Get time elapsed from lass tick.

        Returns:
            a formatted time string
        '''
        elapse = time.time() - self._last
        self._last = time.time()
        return self._elapse_str(elapse)

    def tock(self):
        '''
        Get time elapsed from start or last reset.

        Returns:
            a formatted time string
        '''
        elapse = time.time() - self._start
        return self._elapse_str(elapse)

    def __enter__(self):
        invoke_info = inspect.stack()[1]
        invoke_str = f"[{os.path.basename(invoke_info[1])}:{invoke_info[2]}][{invoke_info[3]}]"
        loggerx.info(f"{invoke_str}: start timing.")
        self.reset()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapse_str = self.tock()
        invoke_info = inspect.stack()[1]
        invoke_str = f"[{os.path.basename(invoke_info[1])}:{invoke_info[2]}][{invoke_info[3]}]"
        loggerx.info(f"{invoke_str}: time elapsed: {elapse_str}")

    def _elapse_str(self, elapse):
        second = round(elapse % 60, ndigits=2)
        minute = int((elapse // 60) % 60)
        hour = int(elapse // 3600)
        elapse_str = self.format_str % (hour, minute, second)
        return elapse_str


def current_datetime(date_sep='-', time_sep=':', date_time_sep='_'):
    """
    Helper function to show current date and time

    Returns:
        datetime_str: string shows current date and time
    """
    format_str = '%Y{0}%m{0}%d{1}%H{2}%M{2}%S'.format(date_sep, date_time_sep, time_sep)
    datetime_str = datetime.datetime.strftime(datetime.datetime.now(), format_str)
    return datetime_str
