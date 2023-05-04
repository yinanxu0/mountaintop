from collections import namedtuple
from typing import Dict
from tabulate import tabulate


from mountaintop.core.internal.timing import Timer, current_datetime
from mountaintop.runx.utils import RunningAverager
from mountaintop.runx.logx import loggerx


class EpochMonitor(object):
    def __init__(self, track_metric=None, higher_better=False):
        self.track_metric = track_metric
        self.higher_better = higher_better

        self.records = []
        self.epoch = 1
        self.stalled_epoch = 0

        self.timer = Timer()
        self.initialized = False

    def post_init(self, fields):
        self.Record = namedtuple("Record", ['epoch'] + fields + ['time_spent'])
        null_record = [None for _ in self.Record._fields]
        self.current = self.Record(*null_record)
        self.best = self.Record(*null_record)
        self.last = self.Record(*null_record)
        self.initialized = True
    
    def init(self):
        self.timer.reset()
    
    def set_epoch(self, epoch):
        self.epoch = epoch

    def _get_record(self, epoch, time_str, **metrics):
        record = []
        for field in metrics.keys():
            record.append(metrics[field])
        record.insert(0, epoch)
        record.append(time_str)
        record = self.Record(*record)
        return record

    def add(self, train_metrics: Dict, valid_metrics: Dict):
        assert train_metrics.keys() == valid_metrics.keys()
        assert self.track_metric in train_metrics
        metrics = {
            f"train_{self.track_metric}": train_metrics[self.track_metric],
            f"valid_{self.track_metric}": valid_metrics[self.track_metric],
        }
        if not self.initialized:
            self.post_init(fields=list(metrics.keys()))
        
        # Get current epoch results
        current = self._get_record(self.epoch, self.timer.tick(), **metrics)
        self.records.append(current)
        self.last = self.current
        self.current = current
        self.epoch += 1

        # check if updates
        best_track = self.best.__getattribute__(f"valid_{self.track_metric}")
        curr_track = current.__getattribute__(f"valid_{self.track_metric}")
        if self.higher_better:
            is_better = not best_track or curr_track > best_track
        else:
            is_better = not best_track or curr_track < best_track
        if is_better:
            self.best = current
            self.stalled_epoch = 0
        else:
            self.stalled_epoch += 1

        table_entries = [
            ['name'] + list(self.Record._fields),
            ['current'] + list(self.current),
            ['best'] + list(self.best),
            ['last'] + list(self.last)
        ]
        # transpose
        epoch_result = tabulate(table_entries[1:], table_entries[0], "pipe", floatfmt=".4f", stralign="right")
        loggerx.info(f"Epoch result: \n{epoch_result}")
        
        return self.stalled_epoch

    def to_file(self, filename):
        with open(filename, 'w') as fout:
            fout.write('=' * 20 + ' ' + current_datetime() + ' ' + '=' * 20 + '\n')
            fout.write(tabulate(self.records, self.Record._fields, floatfmt=".4f") + '\n')


class StepMonitor(object):
    def __init__(self, global_step=0, moving_avg=True, log_interval=100):
        self.records = []
        
        self.epoch = -1
        self.global_step = global_step
        self.log_interval = log_interval
        self.local_step = 0
        self.local_max_step = -1
        
        self.moving_avg = moving_avg
        self.timer = Timer()
        
        self.initialized = False
        self.run_avg = None
        self.Record = None
        
    def post_init(self, fields):
        self.Record = namedtuple("Record", ['global_step'] + fields)
        self.run_avg = RunningAverager(fields, inits=[0.0 for _ in fields])
        self.initialized = True

    def _get_record(self, global_step, **metrics):
        record = []
        for field in metrics.keys():
            record.append(metrics[field])
        record.insert(0, global_step)
        record = self.Record(*record)
        return record

    def _desc_str(self, **metrics):
        desc_elements = []
        for field in metrics.keys():
            desc_elements.append(f'{field}: {metrics[field]:.4f}')
        desc = f'At epoch {self.epoch} step {self.local_step}/{self.local_max_step}: '
        desc += ', '.join(desc_elements)
        desc += f', time elapse: {self.timer.tock()}'
        return desc

    def epoch_init(self):
        if self.local_max_step <= 0:
            self.local_max_step = self.local_step
        self.local_step = 0
        self.timer.reset()
        if self.initialized:
            self.run_avg.reset()
    
    def epoch_result(self):
        return self.run_avg.result._asdict()

    def add(self, epoch, **metrics):
        if self.epoch != epoch:
            self.epoch = epoch
            self.epoch_init()
        
        if not self.initialized:
            self.post_init(fields=list(metrics.keys()))
        
        self.global_step += 1
        self.local_step += 1
        self.run_avg.update(**metrics)
        if self.moving_avg:
            desc = self._desc_str(**self.run_avg.local_result._asdict())
        else:
            desc = self._desc_str(**metrics)
            
        if self.global_step % self.log_interval == 0:
            self.records.append(self._get_record(self.global_step, **self.run_avg.local_result._asdict()))
        
        if self.local_step == 1 or self.local_step % self.log_interval == 0 or self.local_step == self.local_max_step:
            loggerx.info(desc)
            self.run_avg.local_reset()
        else:
            loggerx.debug(desc)

    def to_file(self, filename):
        with open(filename, 'w') as fout:
            fout.write('=' * 20 + ' ' + current_datetime() + ' ' + '=' * 20 + '\n')
            fout.write(tabulate(self.records, self.Record._fields) + '\n')


class ProgressMonitor(object):
    def __init__(self, fields=[], max_step=-1):
        self.fields = fields

        self.local_step = 0
        self.check_step = 0
        self.local_max_step = max_step
        self.timer = Timer()


    def _desc_str(self, info):
        desc = f'Process {info}, time elapse: {self.timer.tock()}'
        return desc


    def epoch_init(self, check_step=1000):
        if self.local_max_step <= 0:
            self.local_max_step = self.local_step
        self.local_step = 0
        self.timer.reset()
        self.check_step = check_step


    def add(self, step=1):
        self.local_step += step
        info = f'at step {self.local_step}/{self.local_max_step}'
        desc = self._desc_str(info)
        if self.local_step == 1 or self.local_step % self.check_step == 0 or self.local_step == self.local_max_step:
            loggerx.info(desc)


    def epoch_finish(self, **fields):
        info1 = f'at step {self.local_step}/{self.local_max_step}'
        info_elems = []
        for field in self.fields:
            if field not in fields:
                raise KeyError(f"Field specified in monitor not provided: '{field}'")
            else:
                info_elems.append(f'{field}: {fields[field]:.4f}')
        info2 = "finished. "
        if len(info_elems) > 0:
            info2 += "Result: " + ', '.join(info_elems)
        for info in [info1, info2]:
            desc = self._desc_str(info)
            loggerx.info(desc)

