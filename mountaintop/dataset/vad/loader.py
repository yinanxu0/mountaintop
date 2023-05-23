import os
import copy
from typing import Optional 
from torch.utils.data import DataLoader
from uphill import SupervisionArray


from mountaintop.dataset.vad.fetcher import (
    get_fetcher, 
    is_train_fetcher,
    PreFetcher, 
    PaddingFetcher
)
from mountaintop.dataset.base import BaseDataset


class VadDatasetLoader:
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
        

    def create_data_loader(
        self, 
        data_path: str, 
        mode: str = 'train', 
        fetcher_configs: Optional[dict] = None,
        num_workers: Optional[int] = None, 
        pin_memory: Optional[bool] = None, 
        prefetch_factor: Optional[int] = None,
        **kwargs
    ):
        """Dataset for loading audio data.

        Attributes::
            data_path: path of data manifests, `uphill.SupervisionArray`
        """
        assert mode in ["train", "valid", "test"]
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"{data_path} not exists! Please double check")
        
        _fetcher_configs = self._fetcher_configs if fetcher_configs is None else fetcher_configs
        fetcher_configs = copy.deepcopy(_fetcher_configs)
        
        num_workers = self._num_workers if num_workers is None else num_workers
        pin_memory = self._pin_memory if pin_memory is None else pin_memory
        prefetch_factor = self._prefetch_factor if prefetch_factor is None else prefetch_factor
        
        if num_workers == 0 and prefetch_factor != 2:
            raise ValueError('prefetch_factor option could only be specified in multiprocessing.'
                             'let num_workers > 0 to enable multiprocessing.')
        assert prefetch_factor > 0
        
        dataset = BaseDataset(data_path, shuffle=(mode=="train"), partition=True)
        if mode != "train":
            train_fetcher = []
            for key in fetcher_configs.keys():
                if is_train_fetcher(key):
                    train_fetcher.append(key)
            for key in train_fetcher:
                fetcher_configs.pop(key, None)
            if "feature" in fetcher_configs:
                fetcher_configs["feature"]["dither"] = 0.0
        dataset = PreFetcher(dataset)
        
        for key, conf in fetcher_configs.items():
            fetcher = get_fetcher(key)
            if fetcher is None:
                continue
            dataset = fetcher(dataset, **conf)
        
        dataset = PaddingFetcher(dataset, mode=mode)
        
        dataset_loader = DataLoader(
            dataset = dataset, 
            batch_size = None, 
            pin_memory = pin_memory,
            num_workers = num_workers, 
            prefetch_factor = prefetch_factor
        )
        
        return dataset_loader