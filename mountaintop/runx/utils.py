import os
import csv
import codecs
import math
import numpy as np
from collections import namedtuple
from tabulate import tabulate


from mountaintop.core.internal.timing import Timer, current_datetime
from mountaintop import loggerx


class RunningAverager(object):
    def __init__(self, fields, inits):
        self.fields = fields
        self.inits = inits

        self.count = 0.0
        self.avgs = list(self.inits)
        self.local_count = 0.0
        self.local_avgs = list(self.inits)
        self.Result = namedtuple("Result", self.fields)

    def update(self, **kargs):
        self.local_count += 1.0
        self.count += 1.0
        for i, field in enumerate(self.fields):
            if field not in kargs:
                raise KeyError(f"Field specified not provided: '{field}'")
            if math.isinf(kargs[field]) or math.isnan(kargs[field]):
                loggerx.debug(f"{field} value is {kargs[field]}, skip")
                kargs[field] = 0.0
            self.local_avgs[i] += (kargs[field] - self.local_avgs[i]) / self.local_count
            self.avgs[i] += (kargs[field] - self.avgs[i]) / self.count

    def local_reset(self):
        self.local_count = 0.0
        self.local_avgs = list(self.inits)
    
    def reset(self):
        self.count = 0.0
        self.avgs = list(self.inits)
        self.local_count = 0.0
        self.local_avgs = list(self.inits)
    
    @property
    def local_result(self):
        return self.Result(*self.local_avgs)

    @property
    def result(self):
        return self.Result(*self.avgs)


class CSVTracker(object):
    def __init__(self, fields, fmts, log_dir, filename):
        self.fields = fields
        self.fmts = fmts
        fout = open(os.path.join(log_dir, '%s.%s.csv' % (filename, current_datetime())), 'w')
        fout.write(codecs.BOM_UTF8)
        self.csv_writer = csv.writer(fout, dialect='excel', delimiter=',', lineterminator='\n')
        self.csv_writer.write_row(['time'] + fields)
        self.timer = Timer()

    def _log_str(self, time, fetch_dict):
        return 'time: %s, ' % time + ', '.join(
                [field + ': ' + fmt % fetch_dict[field] for field, fmt in zip(self.fields, self.fmts)])

    def _csv_record(self, time, fetch_dict):
        return [time] + [fetch_dict[field] for field in self.fields]

    def track(self, fetch_dict):
        time = self.timer.tock()
        loggerx.info(self._log_str(time, fetch_dict))
        self.csv_writer.write_row(self._csv_record(time, fetch_dict))


class Tracker(object):
    def __init__(self, fields, fmts, filename, check_finite=True, timed=True):
        self.fields = fields
        self.fmts = fmts
        self.filename = filename
        self.check_finite = check_finite
        self.timed = timed

        if self.timed:
            self.Record = namedtuple("Record", ['time'] + fields)
            self.timer = Timer()
        else:
            self.Record = namedtuple("Record", fields)
            self.timer = None

        self.is_finite = True
        self.records = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        with open(self.filename, 'w') as fout:
            fout.write('=' * 20 + ' ' + current_datetime() + ' ' + '=' * 20 + '\n')
            fout.write(tabulate(self.records, self.Record._fields) + '\n')

    def track(self, fetch_dict):
        if self.timed:
            time = self.timer.tock()
            loss_str = 'time: %s, ' % time
            record = [time] + [fetch_dict[field] for field in self.fields]
        else:
            loss_str = ''
            record = [fetch_dict[field] for field in self.fields]

        loss_str += ', '.join([field + ': ' + fmt % fetch_dict[field] for field, fmt in zip(self.fields, self.fmts)])
        self.records.append(record)
        loggerx.info(loss_str)

        if self.check_finite:
            for key, val in fetch_dict.items():
                if isinstance(val, np.float32) and not np.isfinite(val):
                    self.is_finite = False
                    loggerx.error("Invalid value: %s: %s" % (key, str(val)))
            return self.is_finite

