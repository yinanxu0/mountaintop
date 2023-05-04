import os
import copy
import functools
from collections import defaultdict
from typing import Dict, Iterable, Optional, Tuple, Mapping, Any

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler as Scheduler
from torch.utils.data import DataLoader, DistributedSampler

try:
    from contextlib import nullcontext
except ImportError:
    from contextlib import contextmanager
    @contextmanager
    def nullcontext():
        yield

from mountaintop.runx.monitor import StepMonitor, EpochMonitor
from mountaintop.core.internal.distribute import (
    get_global_rank, 
    get_local_rank, 
    is_distributed, 
    is_horovod_available,
    scaled_all_reduce
)

from mountaintop.runx.logx import loggerx, to_np
from mountaintop.models.saver import (
    average_state_dict
)
from mountaintop.core.internal.typing import (
    extract_access_model
)


class Trainer(object):
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: Optimizer,
        scheduler: Scheduler = None,
        log_interval: int = 100,
        grad_accum_steps: int = 1,
        update_scheduler_by_epoch: bool = False,
        device: Optional[torch.device or str] = None,
        use_cudnn_benchmark: bool = True,
        use_cuda_nonblocking: bool = False,
        use_sync_bn: bool = True,
        use_horovod: bool = False,
        use_amp: bool = False,
        use_prefetcher: bool = False,
        log_gradient: bool = False,
        keep_old_model: bool = False,
        checkpoint_selector: dict = {},
        grad_clip_norm: Optional[float] = None,
        model_average: bool = True,
        model_average_interval: int = 100,
        # **kwargs,
    ):
        ### setup for model
        if isinstance(model, torch.nn.Module):
            self.model = model
        elif isinstance(model, dict):
            self.model = torch.nn.ModuleDict(model)
        else:
            raise TypeError(
                f'Unknown type for `model`. Expected torch.nn.Module or Dict[str, Module], but got {type(model)}'
            )
        # self.access_model is useful for e.g., checkpointing
        self.access_model = extract_access_model(model=self.model)
        self._average_model = None
        # only master can average model
        self._model_average = model_average and loggerx.is_master
        if self._model_average:
            self._average_model = copy.deepcopy(self.access_model)
        self._average_interval = model_average_interval

        ### setup for metric
        self._eval_metric = checkpoint_selector.get("eval_metric", "loss")
        self._higher_better = checkpoint_selector.get("higher_better", True)
        self._best_metric = -float('inf') if self._higher_better else float('inf')
        
        ### setup for logger
        self._log_interval = log_interval
        self._log_gradient = log_gradient
        self.monitors = defaultdict(lambda: StepMonitor(log_interval=self._log_interval))
        
        ### setup for misc params
        self._step = -1
        self._epoch = -1
        self._is_train = None
        self._use_prefetcher = use_prefetcher
        self._grad_accum_steps = grad_accum_steps
        self._keep_old_model = keep_old_model
        self.grad_clip_norm = grad_clip_norm
        

        if device is None:
            self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        else:
            self.device = device

        ### setup for distributed
        self._use_sync_bn = use_sync_bn
        self.rank = 0
        if is_distributed():
            if self._use_sync_bn:
                loggerx.info('Convert batch norm to sync batch norm')
                self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

            self.rank = get_local_rank()
            torch.cuda.set_device(self.rank)
            if get_global_rank() > 0:
                # to avoid overwriting
                self._verbose = False

        
        if 'cuda' in str(self.device):
            self.model.to(self.device)
            torch.backends.cudnn.benchmark = use_cudnn_benchmark
            self._cuda_nonblocking = use_cuda_nonblocking
            loggerx.debug(
                f'cuda: True, cudnn.benchmark: {use_cudnn_benchmark}, '
                f'cuda.nonblocking: {use_cuda_nonblocking}')
        else:
            self._cuda_nonblocking = False
            # usually, this is not expected
            loggerx.info(
                f'cuda: False (torch.cuda.is_available()={torch.cuda.is_available()})'
            )

        if not use_horovod and is_distributed():
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model,
                device_ids=[self.rank],
                output_device=self.rank,
                find_unused_parameters=True)

        # setup optimizer and scheduler
        self.optimizer = optimizer
        self.scheduler = scheduler
        self._update_scheduler_by_epoch = update_scheduler_by_epoch
        self.set_optimizer()
        self.set_scheduler()

        if use_horovod:
            if not is_horovod_available():
                raise RuntimeError('horovod is not available!')
            import horovod.torch as hvd

            hvd.broadcast_parameters(self.model.state_dict(), root_rank=0)
            hvd.broadcast_optimizer_state(self.optimizer, root_rank=0)
            self.optimizer = hvd.DistributedOptimizer(
                self.optimizer, named_parameters=self.model.named_parameters())

        self._use_amp = use_amp # used for pytorch amp mixed precision training
        self.scaler = None
        if self._use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
            loggerx.info('AMP is activated')
        
        ### kwargs
        # self.kwargs = {}
        # for k, v in kwargs.items():
        #     if hasattr(self, k):
        #         raise AttributeError(f'{self} already has {k}')
        #     if torch.is_tensor(v):
        #         v = v.to(self.device)
        #     if isinstance(v, torch.nn.Module):
        #         v.to(self.device)
        #     self.kwargs[k] = v
    
    @property
    def log_interval(self):
        return self._log_interval

    @property
    def step(self):
        return self._step

    @property
    def epoch(self):
        return self._epoch

    @property
    def best_metric(self):
        return self._best_metric

    @property
    def is_train(self):
        return self._is_train
    
    def iteration(
        self,
        data: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor]:
        with torch.cuda.amp.autocast(
                self._use_amp) if self._use_amp else nullcontext():
            try:
                output, loss, metrics = self.model(**data)
                # collapse all losses if they are scattered on multiple gpus
                loss = scaled_all_reduce([loss])[0]
                for k, m in metrics.items():
                    # TODO: fix metric all reduce
                    if torch.is_tensor(m):
                        m = scaled_all_reduce([m])[0]
                        metrics[k] = m.detach().item()
            except Exception:
                raise ValueError(
                    f'The implemented module = {type(self.access_model)} should return 3-tuples, i.e, output, loss, metrics. '
                )

        is_update_step = ((self.step + 1) % self._grad_accum_steps == 0)
        is_log_step = (self.step + 1) % (self.log_interval *
                                         self._grad_accum_steps) == 0

        if self.is_train:
            # ensure that accumlated gradients are normalized
            _loss = loss
            if self._grad_accum_steps > 1:
                _loss = loss / self._grad_accum_steps

            # back propagation
            if self._use_amp:
                self.scaler.scale(_loss).backward()
                if self.grad_clip_norm is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.grad_clip_norm
                    )
                    
                # self.scaler.step(self.optimizer)
                # self.scaler.update()
            else:
                _loss.backward()
                if self.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.grad_clip_norm
                    )


            # log the layers and layers gradient histogram and distributions
            # NOTE: the visualization step must be called before `zero_grad()`
            if is_log_step and self._log_gradient:
                for tag, value in self.model.named_parameters():
                    tag = tag.replace('.', '/')
                    if value is not None and value.grad is not None:
                        loggerx.add_histogram('model/' + tag, to_np(value),
                                           self.step)

                        loggerx.add_histogram('model/' + tag + '/grad',
                                           to_np(value.grad), self.step)

            # update the parameters
            if is_update_step:
                if self._use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                elif torch.isfinite(loss):
                    self.optimizer.step()

                # reset gradient
                self.model.zero_grad()
                self.optimizer.zero_grad()

                if (self.scheduler
                        is not None) and (not self._update_scheduler_by_epoch):
                    # # update scheduler step by grad_accum_steps
                    # for _ in range(self._grad_accum_steps):
                    #     self.scheduler.step()
                    self.scheduler.step()

        return output, loss, metrics

    def _loop(
        self,
        data_loader: Iterable or DataLoader,
        # mode: str = 'train',
        **kwargs
    ):
        mode = "train" if self.is_train else 'valid'
        # keep tracking the model's metric
        # avg_metrics = AverageDictMeter()

        for batch_idx, batch_data in enumerate(data_loader):
            
            assert isinstance(batch_data, dict), \
                f"batch_data in data_loader should be dictionary"
            
            # move batch of samples to device
            if 'cuda' in str(self.device):
                for key, tensor in batch_data.items():
                    if isinstance(tensor, torch.Tensor):
                        batch_data[key] = tensor.to(self.device, non_blocking=True)

            if self.is_train:
                self._step += 1

            _, loss, metrics = self.iteration(batch_data)
            if loss is not None and "loss" not in metrics:
                # capture metrics
                metrics.update({'loss': loss})
            
            for key, value in metrics.items():
                if isinstance(value, torch.Tensor):
                     value = value.detach().item()
                metrics[key] = value

            # avg_metrics.update(metrics)

            # emptying the CUDA cache after the first step can reduce 
            # the chance of OOM
            if 'cuda' in str(self.device) and batch_idx == 0:
                torch.cuda.empty_cache()

            if self.is_train:
                loggerx.add_scalar(
                    '%s/learning_rate' % mode,
                    self.scheduler.get_lr()[0],
                    self.step,
                )
                loggerx.metric(mode, metrics, self.step)
                if self._model_average and self.step%self._average_interval==0:
                    self.update_averaged_model()
            self.monitors[mode].add(epoch=self.epoch, **metrics)
        
        return self.monitors[mode].epoch_result()

    def train(self, data_loader: Iterable or DataLoader, **kwargs) -> Dict:
        """Training the model for an epoch.

        :param data_loader:
        :param mode: Name of this loop. Default is `train`. Passed to callbacks.
        """
        self._is_train = True
        self._epoch += 1
        
        loggerx.info("Start training")
        lr = self.optimizer.param_groups[0]['lr']
        loggerx.info(f'Epoch {self.epoch} lr {lr}')

        # Turn on the train mode
        self.model.train()

        avg_metrics = None
        with torch.enable_grad():
            avg_metrics = self._loop(data_loader,  **kwargs)

        if self.scheduler is not None and self._update_scheduler_by_epoch:
            self.scheduler.step()

        # For distributed training, to make shuffling work properly across multiple epochs.
        # Otherwise, the same ordering will be always used.
        if isinstance(data_loader, DataLoader) and isinstance(
                data_loader.sampler, DistributedSampler):
            data_loader.sampler.set_epoch(self.epoch)
        return avg_metrics

    @torch.no_grad()
    def valid(self, data_loader: Iterable or DataLoader, name: str = "valid", **kwargs):
        """Evaluate the model.

        :param data_loader:
        :param mode: Name of this loop. Default is `test`. Passed to callbacks.
        :return:
        """
        loggerx.info("Start validation")
        self._is_train = False
        # Turn on the evaluation mode
        self.model.eval()

        with torch.no_grad():
            eval_metrics = self._loop(data_loader, **kwargs)

        loggerx.metric(name, eval_metrics, self.epoch)

        return eval_metrics

    def finish(self):
        for name in ["train", "valid"]:
            filepath = os.path.join(loggerx.logdir, f"{name}.metrics.csv")
            self.monitors[name].to_file(filepath)
        if hasattr(self, "epoch_monitor"):
            self.epoch_monitor.to_file(os.path.join(loggerx.logdir, "epoch.metrics.csv"))
    
    def run(
        self,
        train_loader: Iterable or DataLoader,
        eval_loaders: Iterable or DataLoader or Dict[str, Iterable or DataLoader],
        num_epochs: int,
    ):
        """Train the model for a given iterations. This module is almost equal
        to :: for ep in range(total_iterations): trainer.train(train_loader)
        for k, v in val_loaders.items(): trainer.test(v, k)

        :param train_loader:
        :param val_loaders:
        :param num_epochs:
        :param start_epoch:
        :return:
        """
        
        self.epoch_monitor = EpochMonitor(
            track_metric=self._eval_metric, 
            higher_better=self._higher_better
        )

        default_valid_key = "valid"
        if not isinstance(eval_loaders, Dict) and \
            (isinstance(eval_loaders, Iterable) or isinstance(eval_loaders, DataLoader)):
            eval_loaders = {default_valid_key: eval_loaders}

        start_epoch = self.epoch
        if num_epochs <= start_epoch:
            loggerx.warning(f"training from epoch {start_epoch} to {num_epochs} not valid")
            return
        
        self.epoch_monitor.set_epoch(self.epoch+1)
        for epoch in range(start_epoch, num_epochs):
            self._epoch = epoch
            # train_loader.set_epoch(epoch)
            train_metrics = self.train(train_loader)
            valid_metrics = {}
            for name, loader in eval_loaders.items():
                valid_metrics[name] = self.valid(loader, name)
                
            stalled_epoch = self.epoch_monitor.add(
                train_metrics=train_metrics, 
                valid_metrics=valid_metrics[default_valid_key]
            )
            if stalled_epoch > 0:
                loggerx.info(f"{stalled_epoch} epochs not improved")
            self.save_checkpoint(valid_metrics[default_valid_key])
        self.finish()

    def state_dict(self) -> Mapping[str, Any]:
        state_dict = {
            '__model': self.access_model.state_dict(),
            '__optim': self.optimizer.state_dict(),
            '__scheduler': self.scheduler.state_dict() if self.scheduler else None,
            '__epoch': self.epoch,
            '__step': self.step,
            '__best_metric': self._best_metric,
            '__update_scheduler_by_epoch': self._update_scheduler_by_epoch,
            '__use_sync_bn': self._use_sync_bn,
            '__use_amp': self._use_amp
        }
        if self._model_average:
            state_dict['__average_model'] = self._average_model.state_dict()
        return state_dict

    def load_state_dict(
        self,
        state_dict: Mapping[str, Any],
        resume: bool = False
    ) -> None:
        log_str = '=> loading state dict with missing keys: {}, unexpected keys: {}'
        assert '__model' in state_dict
        missing, unexpected = self.access_model.load_state_dict(
            state_dict['__model'], 
            strict=False
        )
        if len(missing) > 0 or len(unexpected) > 0:
            loggerx.info(log_str.format(missing, unexpected))
        if self._model_average:
            assert '__average_model' in state_dict
            missing, unexpected = self._average_model.load_state_dict(
                state_dict['__average_model'], 
                strict=False
            )
            if len(missing) > 0 or len(unexpected) > 0:
                loggerx.info(log_str.format(missing, unexpected)) 
        if not resume:
            return
        if '__optim' in state_dict:
            self.optimizer.load_state_dict(state_dict['__optim'])
        if '__scheduler' in state_dict:
            self.scheduler.load_state_dict(state_dict['__scheduler'])
        if '__update_scheduler_by_epoch' in state_dict:
            self._update_scheduler_by_epoch = state_dict['__update_scheduler_by_epoch']
        if '__use_sync_bn' in state_dict:
            self._use_sync_bn = state_dict['__use_sync_bn']
        if '__use_amp' in state_dict:
            self._use_amp = state_dict['__use_amp']
        if '__best_metric' in state_dict:
            self._best_metric = state_dict['__best_metric']
        
        if '__epoch' in state_dict:
            self._epoch = state_dict['__epoch']
            if self._update_scheduler_by_epoch:
                self.scheduler.last_epoch = self.epoch
        if '__step' in state_dict:
            self._step = state_dict['__step']
            if not self._update_scheduler_by_epoch:
                self.scheduler.last_epoch = self.step
        

    def save_checkpoint(self, eval_metrics, *arg, **kwargs):
        """checkpoint saving."""
        if self._eval_metric not in eval_metrics:
            raise ValueError(
                f"The model's metric {self._eval_metric} is not available!")

        metric = eval_metrics[self._eval_metric]
        self._best_metric = (
            max(self.best_metric, metric) if self._higher_better else min(
                self.best_metric, metric))

        loggerx.save_model(
            self.state_dict(),
            metric=metric,
            epoch=self.epoch,
            higher_better=self._higher_better,
            delete_old=not self._keep_old_model
        )

    def load_checkpoint(
        self,
        filepath: str,
        resume: bool = False,
        **kwargs
    ):
        """Restore a model and return a dict with any meta data included in the
        snapshot."""
        if os.path.isfile(filepath):
            checkpoint = torch.load(
                filepath, 
                map_location=torch.device('cpu')
            )
            info_str = f"=> loaded checkpoint '{filepath}'"
            info_str += f" (epoch {checkpoint['__epoch']})" if "__epoch" in checkpoint else ""
            loggerx.info(info_str)
            self.load_state_dict(checkpoint, resume=resume)
        else:
            loggerx.warning(f"=> no checkpoint found at '{filepath}'")
            raise FileNotFoundError(f'checkpoint file {filepath} not found!')

    def set_optimizer(self) -> None:
        """Set optimizer(s) for model(s).

        You can override as::
            class YourTrainer(TrainerBase):
                def set_optimizer(self):
                    self.optimizer = torch.optim.SGD(self.model.parameters())
        :return:
        """
        optimizer = self.optimizer
        if isinstance(optimizer, Optimizer) or optimizer is None:
            self.optimizer = optimizer

        elif isinstance(optimizer, functools.partial):
            if not issubclass(optimizer.func, Optimizer):
                raise TypeError(
                    f'`optimizer.func` is expected to be subclass of `Optimizer`'
                    f' but got {type(optimizer.func)}')

            grouped_parameters = self.model.parameters()
            if hasattr(self.model, 'optimizer_grouped_parameters'):
                grouped_parameters = self.model.optimizer_grouped_parameters

            self.optimizer = optimizer(grouped_parameters)

        # elif isinstance(optimizer, dict):
        #     if not isinstance(self.model, torch.nn.ModuleDict):
        #         raise TypeError(
        #             'When `optimizer` is `dict`, `model` also needs to be `dict` or `torch.nn.ModuleDict`'
        #         )

        #     if isinstance(list(optimizer.values())[0], functools.partial):
        #         optimizer = {
        #             k: v(self.model[k].parameters())
        #             for k, v in optimizer.items() if v is not None
        #         }
        #     self.optimizer = StepDict(Optimizer, **optimizer)

        else:
            raise TypeError(
                f'Unexpected type {type(optimizer)} for `optimizer`')

    def set_scheduler(self) -> None:
        """Set scheduler(s) for optimizer(s).

        You can override as ::
            class YourTrainer(TrainerBase):
                def set_scheduler(self):
                    self.scheduler = torch.optim.lr_scheduler.Foo(self.optimizer)
        :return:
        """
        scheduler = self.scheduler
        if scheduler is not None and self.optimizer is None:
            raise TypeError('Optimizer is not set, so scheduler cannot be set')

        if isinstance(scheduler, Scheduler) or scheduler is None:
            self.scheduler = scheduler

        elif isinstance(scheduler, functools.partial):
            self.scheduler = scheduler(self.optimizer)

        # elif isinstance(scheduler, dict):
        #     if not isinstance(self.optimizer, StepDict):
        #         raise TypeError(
        #             'When `scheduler` is `dict`, `optimizer` is also needs to be `dict`'
        #         )

        #     _scheduler = {}
        #     for k, v in scheduler.items():
        #         if isinstance(v, functools.partial):
        #             v = v(self.optimizer[k])
        #         _scheduler[k] = v
        #     self.scheduler = StepDict(Scheduler, **_scheduler)

        else:
            raise TypeError(
                f'Unexpected type {type(scheduler)} for `scheduler`')

    def update_averaged_model(self):
        if not loggerx.is_master:
            # when using NCCL without GPU, local rank would be -1
            return
        if self.step <= 0:
            return
        weight_current = self._average_interval / self.step
        weight_average = 1 - weight_current
        avg_state_dict = self._average_model.state_dict()
        cur_state_dict = self.access_model.state_dict()
        average_state_dict(
            state_dict_1=avg_state_dict,
            state_dict_2=cur_state_dict,
            weight_1=weight_average,
            weight_2=weight_current,
        )