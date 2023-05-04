import functools
import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, IterableDataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler
from uphill import SupervisionArray


from mountaintop.dataset.sampler import DistributedSampler as MyDistributedSampler
from mountaintop.core.internal.distribute import get_global_rank, get_world_size, is_distributed


class BaseDataset(IterableDataset):
    def __init__(self, data_path, shuffle=False, partition=True, *args, **kwargs):
        self._data_path = data_path
        self._data_loaded = False
        self._sampler = MyDistributedSampler(shuffle=shuffle, partition=partition)
        self._shuffle = shuffle
        self._partition = partition

    def _load_data(self):
        self._sampler.update()
        if self._partition:
            partition = f"{self._sampler.rank*self._sampler.num_workers + self._sampler.worker_id}/{self._sampler.num_workers*self._sampler.world_size}"
        else:
            partition = f"{self._sampler.worker_id}/{self._sampler.num_workers}"
        self._data = SupervisionArray.from_file(self._data_path, partition=partition, drop_last=True)
        self._data_loaded = True

    def __iter__(self):
        if not self._data_loaded:
            self._load_data()
        if self._shuffle:
            self._data.shuffle()
        for data in self._data:
            yield data


class BaseDatasetLoader(object):
    """Generates and stores PyTorch DataLoader objects for the train, dev and
    test datasets."""

    def __init__(
        self, 
        num_workers: int,
        fetcher_configs, 
        pin_memory: bool = False,
        prefetch_factor: int = 2,
        **kwargs
    ):
        if num_workers == 0 and prefetch_factor != 2:
            raise ValueError('prefetch_factor option could only be specified in multiprocessing.'
                             'let num_workers > 0 to enable multiprocessing.')
        assert prefetch_factor > 0
        
        self._fetcher_configs = fetcher_configs
        
        self._num_workers = num_workers
        self._pin_memory = pin_memory
        self._prefetch_factor = prefetch_factor


    def create_dataset(self, data_path: str, mode: str = 'train', **kwargs):
        raise NotImplementedError

    def create_data_loader(
        self,
        data_path: str,
        batch_size: int,
        mode: str = 'train',
        pin_memory: bool = True,
        num_workers: int = 0,
        start_epoch: bool = 0,
        **kwargs
    ):
        """Wraps a PyTorch Dataset with a DataLoader.

        :param dataset: Dataset to be wrapped.
        :type dataset: Dataset
        :param sampler: PyTorch sampler used to pick samples in a batch.
        :type sampler: Sampler
        :param batch_size: Number of samples in the batch.
        :param num_workers: number of workers to use for the DataLoader
        :type num_workers: int
        :param pin_memory: argument for Data Loader to use page-locked memory for faster transfer of data to GPU
        :type pin_memory: bool
        :return: A DataLoader that wraps the input Dataset.
        """
        dataset = self.create_dataset(data_path, mode, **kwargs)
        is_train = True if mode == 'train' else False

        sampler = None
        drop_last = False
        if is_train and hasattr(dataset, '__len__'):
            if is_distributed():
                sampler_kwargs = dict(
                    num_replicas=get_world_size(),
                    rank=get_global_rank(),
                    shuffle=True if is_train else False)
                sampler = DistributedSampler(dataset, **sampler_kwargs)
                drop_last = True if is_train else False

                # In distributed mode, calling the :meth`set_epoch(epoch) <set_epoch>` method
                # at the beginning of each epoch **before** creating the `DataLoader` iterator is necessary
                # to make shuffling work properly across multiple epochs.
                #  Otherwise, the same ordering will be always used.
                sampler.set_epoch(start_epoch)
            elif not isinstance(dataset, IterableDataset):
                sampler = RandomSampler(dataset, True)

        # NOTE: `IterableDataset` does not work with sampler
        if isinstance(dataset, IterableDataset):
            sampler = None

        loader_kwargs = dict()
        # When supported, use 'forkserver' to spawn dataloader workers instead of 'fork' to prevent
        # issues with Infiniband implementations that are not fork-safe
        if (num_workers > 0 and hasattr(mp, '_supports_context')
                and mp._supports_context
                and 'forkserver' in mp.get_all_start_methods()):
            loader_kwargs['multiprocessing_context'] = 'forkserver'

        return DataLoader(
            dataset,
            sampler=sampler,
            batch_size=batch_size,
            pin_memory=self._pin_memory,
            num_workers=self._num_workers,
            worker_init_fn=functools.partial(self.worker_init_fn),
            drop_last=drop_last,
            **loader_kwargs)
