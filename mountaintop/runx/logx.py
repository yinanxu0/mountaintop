"""
Partially borrow code from https://github.com/NVIDIA/runx/blob/master/runx/logx.py
"""
from collections import defaultdict
from contextlib import contextmanager
import copy
import logging
from shutil import copyfile
from pathlib import Path
from typing import Any, Optional, Union
from termcolor import colored
import csv
import re
import os
import shlex
import subprocess
import time
import torch
try:
    from torch.utils.tensorboard import SummaryWriter
except ModuleNotFoundError:
    from tensorboardX import SummaryWriter


trn_names = ('trn', 'train', 'training')
val_names = ('val', 'valid', 'validate', 'validation')
test_names = ('test')


def get_logger(
    filepath: Optional[Union[Path, str]] = None,
    detail_level: int = 1
) -> logging.Logger:
    logger = logging.getLogger("mountaintop")
    logger.setLevel(logging.DEBUG)
    # remove existed handler
    if len(logger.handlers) > 0:
        for handler in logger.handlers:
            logger.removeHandler(handler)
    logger.addHandler(
        stream_handler(
            level=logging.INFO,
            detail_level=detail_level
        )
    )
    if filepath is not None:
        logger.addHandler(
            file_handler(
                filepath=filepath,
                level=logging.DEBUG,
                detail_level=detail_level
            )
        )
    logger.addFilter(NoIdentifierFilter())
    return logger


###############
# Handler
###############
def stream_handler(level=logging.INFO, detail_level: int = 1):
    handler = logging.StreamHandler()
    handler.setLevel(level)
    handler.setFormatter(ColoredFormatter(to_file=False, detail_level=detail_level))
    return handler

def file_handler(filepath, level=logging.INFO, detail_level: int = 1):
    handler = logging.FileHandler(filepath)
    handler.setLevel(level)
    handler.setFormatter(ColoredFormatter(to_file=True, detail_level=detail_level))
    return handler


###############
# Formatter
###############
class ColoredFormatter(logging.Formatter):
    BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE = range(8)
    COLOR_SEQ = "\033[1;%dm"
    BOLD_SEQ = "\033[1m"
    RESET_SEQ = "\033[0m"

    def __init__(
        self,
        to_file: bool = False,
        detail_level: int = 1
    ):
        self.to_file = to_file
        self.colors = {
            'WARNING': self.YELLOW,
            'INFO': self.CYAN,
            'DEBUG': self.WHITE,
            'CRITICAL': self.YELLOW,
            'ERROR': self.RED
        }
        assert detail_level in [1,2,3], f"`detail_level` shoul be one of 1, 2, 3, {detail_level} not allowed"
        if detail_level == 1:
            msg = "$BOLD%(levelname)8s$RESET: %(message)s"
        elif detail_level == 2:
            msg = "$BOLD%(levelname)8s:[%(filename)s@%(lineno)d:%(funcName)s]$RESET: %(message)s"
        else:
            msg = "%(asctime)s $BOLD%(levelname)8s:[%(filename)s@%(lineno)d:%(funcName)s]$RESET: %(message)s"
        if self.to_file:
            msg = msg.replace("$RESET", '').replace("$BOLD", '')
        else:
            msg = msg.replace("$RESET", self.RESET_SEQ).replace("$BOLD", self.BOLD_SEQ)
        logging.Formatter.__init__(self, msg)

    def format(self, record):
        format_record = record
        levelname = record.levelname
        if not self.to_file and levelname in self.colors:
            levelname_color = self.COLOR_SEQ % (30 + self.colors[levelname]) + levelname
            format_record = copy.copy(record)
            format_record.levelname = levelname_color
        return logging.Formatter.format(self, format_record)


###############
# Filter
###############
class TFlogFilter(logging.Filter):
    def filter(self, record):
        if "tensorflow" in record.pathname or "tf" in record.pathname:
            return False
        else:
            return True

class NoIdentifierFilter(logging.Filter):
    def filter(self, record):
        return not record.getMessage().startswith("initializing identifier")


def is_list(x):
    return isinstance(x, (list, tuple))


def get_gpu_utilization_pct():
    '''
    Use nvidia-smi to capture the GPU utilization, which is reported as an
    integer in range 0-100.
    '''
    util = subprocess.check_output(
        shlex.split('nvidia-smi --query-gpu="utilization.gpu" '
                    '--format=csv,noheader,nounits -i 0'))
    util = util.decode('utf-8')
    util = util.replace('\n', '')
    return int(util)


def to_np(x):
    """https://github.com/yunjey/pytorch-
    tutorial/blob/master/tutorials/04-utils/tensorboard/main.py#L20.

    :param x:
    :return:
    """
    return x.data.cpu().numpy()


class _CallableProxy:
    def __init__(self, real_callable, post_hook=None):
        self.real_callable = real_callable
        self.post_hook = post_hook

    def __call__(self, *args, **kwargs):
        ret_val = self.real_callable(*args, **kwargs)

        if self.post_hook is not None:
            self.post_hook()

        return ret_val


class ConditionalProxy:
    """
    This object can be used to serve as a proxy on an object where we want to
    forward all function calls along to the dependent object, but only when
    some condition is true. For example, the primary use case for this object
    is to deal with that fact that in a distributed training job, we only want
    to manage artifacts (checkpoints, logs, TB summaries) on the rank-0
    process.

    So, let's say that we have this class:
    ```
    class Foo:
        def bar(self, val):
            pass

        def baz(self, val1, val2):
            pass
    ```

    and we wrap it with this object:
    ```
    proxy = ConditionalProxy(Foo(), rank == 0)
    proxy.bar(42)
    proxy.baz(10, 20)
    proxy.some_other_function('darn it')  # Throws an exception because `Foo`
                                          # doesn't have an implementation for
                                          # this.
    ```

    In this case, if `rank == 0`, then we will end up calling `Foo.bar` and
    `Foo.baz`.
    If `rank != 0`, then the calls will be ignored.

    In addition to the basic usage, you can also add a `post_hook` to the
    proxy, which is a callable that takes no arguments. The proxy will call
    that function after each function call made through the proxy, but only
    when `condition == True`.
    """

    def __init__(self, real_object, condition, post_hook=None):
        self.real_object = real_object
        self.condition = condition
        self.post_hook = post_hook

    @staticmethod
    def _throw_away(*args, **kwargs):
        pass

    def __getattr__(self, name):
        if not self.condition:
            # When `self.condition == False`, then we want to return a function
            # that can take any form of arguments, and does nothing. This works
            # under the assumption that the only API interface for the
            # dependent object is function, e.g. this would be awkward if the
            # caller was trying to access a member variable.
            return ConditionalProxy._throw_away

        real_fn = getattr(self.real_object, name)

        # Wrap the return function in a `_CallableProxy` so that we can
        # invoke the `self.post_hook`, if specified, after the real function
        # executes.
        return _CallableProxy(real_fn, self.post_hook)


class LoggerX(object):
    def __init__(self, rank=0):
        self._initialized = False
        self.detail_level = int(os.environ.get("DETAIL_LEVEL", 1))
        self.logger = get_logger(filepath=None, detail_level=self.detail_level)

    def initialize(
        self,
        logdir: Union[str, Path] = "/tmp",
        to_file: bool = True,
        hparams=None,
        tensorboard=False,
        no_timestamp=False,
        global_rank=0,
        eager_flush=True,
        detail_level: Optional[int] = None
    ):
        '''
        Initialize LoggerX

        inputs
        - logdir - where to write logfiles
        - to_file - whether to write log to files
        - tensorboard - whether to write to tensorboard file
        - global_rank - must set this if using distributed training, so we only
          log from rank 0
        - coolname - generate a unique directory name underneath logdir, else
          use logdir as output directory
        - hparams - only use if not launching jobs with runx, which also saves
          the hparams.
        - eager_flush - call `flush` after every tensorboard write
        - detail_level - control log record details
        '''
        if self._initialized:
            self.logger.warning(
                "loggerx has been initialized, cannot initialize again",
                stacklevel=3
            )
            return
        self.is_master = (global_rank == 0)
        if not self.is_master:
            return

        self.logdir = logdir if isinstance(logdir, Path) else Path(logdir)
        log_path = None
        if to_file:
            # confirm target log directory exists
            if not self.logdir.exists():
                self.logdir.mkdir(parents=True, exist_ok=True)
            from mountaintop.core.internal.timing import current_datetime
            log_path = self.logdir / f'running.{current_datetime()}.log'
        self.detail_level = self.detail_level if detail_level is None else detail_level
        self.logger = get_logger(filepath=log_path, detail_level=self.detail_level)

        if hparams is not None and len(hparams) > 0 and self.is_master:
            from mountaintop.utils.yaml import save_yaml
            hparams_path = self.logdir / "hparams.yaml"
            save_yaml(hparams_path, hparams)

        # Tensorboard file
        self.tb_writer = None
        if self.is_master and tensorboard:
            self.tb_writer = SummaryWriter(
                log_dir=self.logdir,
                flush_secs=1
            )

        self.eager_flush = eager_flush

        # This allows us to use the tensorboard with automatic checking of both
        # the `tensorboard` condition, as well as ensuring writes only happen
        # on is_master. Any function supported by `SummaryWriter` is supported by
        # `ConditionalProxy`. Additionally, flush will be called after any call
        # to this.
        self.tensorboard = ConditionalProxy(
            self.tb_writer,
            tensorboard and self.is_master,
            post_hook=self._flush_tensorboard,
        )

        # Metrics file
        metrics_fn = self.logdir / 'metrics.csv'
        self.metrics_fp = open(metrics_fn, mode='a+')
        self.metrics_writer = csv.writer(self.metrics_fp, delimiter=',')

        # save metric
        self.save_metric = None
        self.best_metric = None
        self.save_ckpt_path = None
        # Find the existing best checkpoint, and update `best_metric`,
        # if available
        self.best_ckpt_path = self.get_best_checkpoint() or None
        if self.best_ckpt_path:
            _, best_chk = self.load_model(self.best_ckpt_path)
            self.best_metric = best_chk.get('__metric', None)
        self.epoch = defaultdict(lambda: 0)
        self.no_timestamp = no_timestamp

        # Initial timestamp, so that epoch time calculation is correct
        phase = 'start'
        csv_line = [phase]

        # add epoch/iter
        csv_line.append(f'{phase}/step')
        csv_line.append(0)

        # add timestamp
        if not self.no_timestamp:
            # this feature is useful for testing
            csv_line.append('timestamp')
            csv_line.append(time.time())

        self.metrics_writer.writerow(csv_line)
        self.metrics_fp.flush()

        self._initialized = True

    def __del__(self):
        if self._initialized and self.is_master:
            self.metrics_fp.close()

    def _check_master(self):
        if not hasattr(self, 'is_master') or not self.is_master:
            return False
        return True

    def _check_msg(self, msg: Any=""):
        if msg is None or len(msg) == 0:
            self.logger.warning(
                "empty message for logger is not recommended, skip",
                stacklevel=3
            )
            return False
        return True

    #### logger functions
    def debug(self, msg: Any, stacklevel=2, *args, **kwargs):
        if not self._check_master() or not self._check_msg(msg):
            return
        self.logger.debug(msg, stacklevel=stacklevel, *args, **kwargs)

    def info(self, msg: Any, stacklevel=2, *args, **kwargs):
        if not self._check_master() or not self._check_msg(msg):
            return
        self.logger.info(msg, stacklevel=stacklevel, *args, **kwargs)
    
    def warning(self, msg: Any, stacklevel=2, *args, **kwargs):
        if not self._check_master() or not self._check_msg(msg):
            return
        self.logger.warning(msg, stacklevel=stacklevel, *args, **kwargs)
    
    def error(self, msg: Any, stacklevel=2, *args, **kwargs):
        if not self._check_master() or not self._check_msg(msg):
            return
        self.logger.error(msg, stacklevel=stacklevel, *args, **kwargs)
    
    def critical(self, msg: Any, stacklevel=2, *args, **kwargs):
        if not self._check_master() or not self._check_msg(msg):
            return
        self.logger.critical(msg, stacklevel=stacklevel, *args, **kwargs)
    
    def add_image(self, path, img, step=None):
        '''
        Write an image to the tensorboard file
        '''
        if not self._check_master():
            return
        self.tensorboard.add_image(path, img, step)

    def add_scalar(self, name, val, idx):
        '''
        Write a scalar to the tensorboard file
        '''
        if not self._check_master():
            return
        self.tensorboard.add_scalar(name, val, idx)

    def _flush_tensorboard(self):
        if self.eager_flush and self.tb_writer is not None:
            self.tb_writer.flush()

    @contextmanager
    def suspend_flush(self, flush_at_end=True):
        prev_flush = self.eager_flush
        self.eager_flush = False
        yield
        self.eager_flush = prev_flush
        if flush_at_end:
            self._flush_tensorboard()
    
    def _canonical_phase(self, phase):
        # define canonical phase
        if phase in trn_names:
            canonical_phase = 'train'
        elif phase in val_names:
            canonical_phase = 'valid'
        else:
            raise(f'expected phase to be one of {val_names} {trn_names}')
        return canonical_phase

    def metric(self, phase, metrics, epoch=None):
        """Record train/val metrics. This serves the dual-purpose to write these
        metrics to both a tensorboard file and a csv file, for each parsing by
        sumx.

        Arguments:
            phase: 'train' or 'val'. sumx will only summarize val metrics.
            metrics: dictionary of metrics to record
            global_step: (optional) epoch or iteration number
        """
        if not self._check_master():
            return
        canonical_phase = self._canonical_phase(phase=phase)

        if epoch is not None:
            self.epoch[canonical_phase] = epoch

        # Record metrics to csv file
        csv_line = [canonical_phase]
        for key, value in metrics.items():
            csv_line.append(key)
            csv_line.append(f"{value:.4f}")

        # add epoch/iter
        csv_line.append('epoch')
        csv_line.append(self.epoch[canonical_phase])

        # add timestamp
        if not self.no_timestamp:
            # this feature is useful for testing
            csv_line.append('timestamp')
            csv_line.append(f"{time.time():.4f}")

        # To save a bit of disk space, only save validation metrics
        if canonical_phase == 'valid':
            self.metrics_writer.writerow(csv_line)
            self.metrics_fp.flush()

        # Write updates to tensorboard file
        with self.suspend_flush():
            for key, value in metrics.items():
                self.add_scalar(
                    f'{canonical_phase}/{key}',
                    value,
                    self.epoch[canonical_phase]
                )

        # if no step, then keep track of it automatically
        if epoch is None:
            self.epoch[canonical_phase] += 1

    @staticmethod
    def is_better(save_metric, best_metric, higher_better):
        return best_metric is None or \
            (higher_better and save_metric > best_metric) or \
            (not higher_better and save_metric < best_metric)

    def save_model(self, save_dict, metric, epoch, higher_better=True,
                   delete_old=True):
        """Saves a model to disk. Keeps a separate copy of latest and best models.

        Arguments:
            save_dict: dictionary to save to checkpoint
            epoch: epoch number, used to name checkpoint
            metric: metric value to be used to evaluate whether this is the
                    best result
            higher_better: True if higher valued metric is better, False
                    otherwise
            delete_old: Delete prior 'lastest' checkpoints. By setting to
                    false, you'll get a checkpoint saved every time this
                    function is called.
        """
        if not self._check_master():
            return
        
        if "__metric" not in save_dict:
            save_dict["__metric"] = metric

        if self.save_ckpt_path is not None and self.save_ckpt_path.exists() and delete_old:
            self.save_ckpt_path.unlink(missing_ok=True)
        # Save out current model
        self.save_ckpt_path = self.logdir / f'ckpt_ep{epoch:04d}.pth'
        self.info(f"save checkpoint to {self.save_ckpt_path}")
        torch.save(save_dict, self.save_ckpt_path)
        self.save_metric = metric

        is_better = self.is_better(self.save_metric, self.best_metric,
                                   higher_better)
        
        if is_better:
            if self.best_ckpt_path is not None and self.best_ckpt_path.exists():
                self.debug(f"remove last best checkpoint {self.best_ckpt_path}")
                self.best_ckpt_path.unlink(missing_ok=True)
            self.best_ckpt_path = self.logdir / f'best_ckpt_ep{epoch:04d}.pth'
            self.info(f"update best checkpoint to {self.best_ckpt_path}")
            self.best_metric = self.save_metric
            copyfile(self.save_ckpt_path, self.best_ckpt_path)
        return is_better

    def get_best_checkpoint(self):
        """
        Finds the checkpoint in `self.logdir` that is considered best.

        If, for some reason, there are multiple best checkpoint files, then
        the one with the highest epoch will be preferred.

        Returns:
            None - If there is no best checkpoint file
            path (str) - The full path to the best checkpoint otherwise.
        """
        best_ckpt_pattern = r'^best_ckpt_ep([0-9]+).pth$'
        best_epoch = -1
        best_ckpt_path = None
        for filepath in self.logdir.glob("best*.pth"):
            if filepath.is_dir():
                continue
            match = re.fullmatch(best_ckpt_pattern, filepath.name)
            if match is not None:
                # Extract the epoch number
                epoch = int(match.group(1))
                if epoch > best_epoch:
                    best_epoch = epoch
                    best_ckpt_path = filepath

        return best_ckpt_path

    def load_model(self, path):
        """Restore a model and return a dict with any meta data included in
        the snapshot
        """
        if not path.exists():
            return {}, {}
        checkpoint = torch.load(path, map_location=torch.device('cpu'))
        model_dict = {}
        if "__model" in checkpoint:
            model_dict = checkpoint.pop('__model')
        elif "state_dict" in checkpoint:
            model_dict = checkpoint.pop('state_dict')
        meta = copy.deepcopy(checkpoint)
        return model_dict, meta

    def summary_model(self, model):
        model_parameters = filter(lambda p: p.requires_grad, model.parameters())
        total_params = sum(p.numel() for p in model_parameters)
        self.info(f"Number of parameter = {total_params}")

    def add_histogram(self, name, val, idx):
        """Write a histogram to the tensorboard file."""
        self.tensorboard.add_histogram(name, val, idx)

    def add_hparams(self, hparam_dict, metric_dict):
        self.tensorboard.add_hparams(
            hparam_dict=hparam_dict, metric_dict=metric_dict)


loggerx = LoggerX()
