from collections import defaultdict
from collections.abc import MutableMapping
from copy import deepcopy
import torch


class AverageMeter(object):
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        if torch.is_tensor(val):
            val = val.detach().item()
        self.val = val
        self.sum += val * n
        self.count += n

    @property
    def average(self):
        if self.count == 0: return None
        return self.sum / self.count


class AverageDictMeter(MutableMapping, dict):
    """a grouping of average meters."""
    __slots__ = ['_data']

    def __init__(self):
        super().__init__()
        self._data = defaultdict(AverageMeter)

    def reset(self):
        for k in self._data.keys():
            del self._data[k]

    def update(self, val_dict, n=1):
        for k, v in val_dict.items():
            self._data[k].update(v, n)

    @property
    def average(self):
        return {k: v.average for k, v in self._data.items()}

    def __getitem__(self, item):
        return self._data[item].average

    def __setitem__(self, key, value):
        self._data[key].update(value)

    def __len__(self):
        return len(self._data)

    def __delitem__(self, key):
        del self._data[key]

    def __iter__(self):
        return iter({k: v.average for k, v in self._data.items()})

    def __contains__(self, key):
        return key in self._data

    def items(self):
        return [(k, v.average) for k, v in self._data.items()]

    def keys(self):
        return self._data.keys()

    def values(self):
        return [v.average for v in self._data.values()]


class TensorMap(MutableMapping, dict):
    """dict like object but: stored values can be subscribed and attributed. ::

        >>> m = TensorMap(test="test")
        >>> m.test is m["test"]
    """

    # inherit `dict` to avoid problem with `backward_hook`s
    __default_methods = [
        'update',
        'keys',
        'items',
        'values',
        'clear',
        'copy',
        'get',
        'pop',
        'to',
        'deepcopy',
    ] + __import__('keyword').kwlist
    __slots__ = ['_data']

    def __init__(self, **kwargs):
        super(TensorMap, self).__init__()
        self._data = {}
        if len(kwargs) > 0:
            self._data.update(kwargs)

    def __getattr__(self, item):
        if item in self.__default_methods:
            return getattr(self, item)
        return self._data.get(item)

    def __setattr__(self, key, value):
        if key == '_data':
            # initialization!
            super(TensorMap, self).__setattr__(key, value)
        elif key not in self.__default_methods:
            self._data[key] = value
        else:
            raise KeyError(f'{key} is a method name.')

    def __getitem__(self, item):
        return self.__getattr__(item)

    def __setitem__(self, key, value):
        self.__setattr__(key, value)

    def __len__(self):
        return len(self._data)

    def __delitem__(self, key):
        del self._data[key]

    def __iter__(self):
        return iter(self._data)

    def __repr__(self):
        _str = self.__class__.__name__ + '('
        for k, v in self._data.items():
            _str += f'{k}={str(v)}, '
        # to strip the las ", "
        return _str.strip(', ') + ')'

    def keys(self):
        return self._data.keys()

    def values(self):
        return self._data.values()

    def to(self, device: str, **kwargs):
        """Move stored tensors to a given device."""

        for k, v in self._data.items():
            if isinstance(v, torch.Tensor):
                self._data[k] = v.to(device, **kwargs)
        return self

    def deepcopy(self):
        new = TensorMap()
        for k, v in self._data.items():
            if isinstance(v, torch.Tensor):
                # Only leave tensors support __deepcopy__
                # detach creates a new tensor
                new[k] = v.detach()
            else:
                new[k] = deepcopy(v)
        return new

    def copy(self):
        new = TensorMap()
        new._data = self._data.copy()
        return new


class TensorTuple(tuple):
    """Tuple for tensors."""

    def to(self, *args, **kwargs):
        """Move stored tensors to a given device."""

        return TensorTuple(
            (t.to(*args, **kwargs) for t in self if torch.is_tensor(t)))
