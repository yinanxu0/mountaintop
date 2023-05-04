import torch
import torch.distributed as dist
from torch.utils.data import get_worker_info

from mountaintop import loggerx


class BaseSampler(object):
    def __init__(self, shuffle: bool=True, partition: bool=True):
        self._epoch = -1
        self._shuffle = shuffle
        self._partition = partition

    def set_epoch(self, epoch: int):
        self._epoch = epoch

    def update(self):
        raise NotImplementedError("not implemented...")

    def sample(self, data):
        raise NotImplementedError("not implemented...")


class WorldSampler(BaseSampler):
    def __init__(self, shuffle=True, partition=True):
        super().__init__(shuffle=shuffle, partition=partition)
        self.rank = 0
        self.world_size = 1
        self.update()

    def update(self):
        assert dist.is_available()
        if dist.is_initialized():
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        loggerx.debug(f"dist info, rank={self.rank}, world_size={self.world_size}")

    def sample(self, data):
        """ Sample data according to rank/world_size

            Args:
                data: data manifests, `uphill.SupervisionArray`

            Returns:
                subset of data manifests after sample, `uphill.SupervisionArray`
        """
        if self._shuffle:
            data = data.shuffle()
        if self._partition and self.world_size > 1:
            subset = data.split(num_splits=self.world_size)[self.rank]
            loggerx.debug(f"world partition data, length={len(data)} => length={len(subset)}, rank={self.rank}, world_size={self.world_size}")
        else:
            subset = data
        return subset


class WorkerSampler(BaseSampler):
    def __init__(self, shuffle=True, partition=True):
        super().__init__(shuffle=shuffle, partition=partition)
        self.worker_id = 0
        self.num_workers = 1
        self.update()

    def update(self):
        worker_info = get_worker_info()
        if worker_info is not None:
            self.worker_id = worker_info.id
            self.num_workers = worker_info.num_workers
        loggerx.debug(f"worker info, worker_id={self.worker_id}, num_workers={self.num_workers}")

    def sample(self, data):
        """ Sample data according to num_workers

            Args:
                data: data manifests, `uphill.SupervisionArray`

            Returns:
                subset of data manifests after sample, `uphill.SupervisionArray`
        """
        if self._shuffle:
            data = data.shuffle()
        if self._partition and self.num_workers > 1:
            subset = data.split(num_splits=self.num_workers)[self.worker_id]
            loggerx.debug(f"worker partition data, length={len(data)} => length={len(subset)}, worker_id={self.worker_id}, num_workers={self.num_workers}")
        else:
            subset = data
        return subset


class DistributedSampler(BaseSampler):
    def __init__(self, shuffle=True, partition=True):
        super().__init__(shuffle=shuffle, partition=partition)
        self._world_sampler = WorldSampler(shuffle=shuffle, partition=partition)
        self._worker_sampler = WorkerSampler(shuffle=shuffle, partition=partition)
        self.update()

    @property
    def rank(self):
        return self._world_sampler.rank

    @property
    def world_size(self):
        return self._world_sampler.world_size

    @property
    def worker_id(self):
        return self._worker_sampler.worker_id

    @property
    def num_workers(self):
        return self._worker_sampler.num_workers

    def update(self):
        self._world_sampler.update()
        self._worker_sampler.update()
        
    def sample(self, data):
        """ Sample data according to rank/world_size/num_workers

            Args:
                data: data manifests, `uphill.SupervisionArray`

            Returns:
                subset of data manifests after sample, `uphill.SupervisionArray`
        """
        data = self._world_sampler.sample(data)
        data = self._worker_sampler.sample(data)
        return data

