"""utils for distributed training."""
import os
import builtins
import importlib.util
from typing import Optional
import dataclasses
import numpy as np
import torch
from torch import distributed
from torch.cuda import device_count


from mountaintop.runx.logx import loggerx

# IS_DISTRIBUTED is used to handle horovod
IS_DISTRIBUTED_HOROVOD = False


# distributed
def init_distributed(
    use_horovod: bool = False,
    backend: Optional[str] = None,
    init_method: Optional[str] = None,
):
    """Simple initializer for distributed training.

    :param use_horovod: If use horovod as distributed backend
    :param backend: backend of torch.distributed.init_process_group
    :param init_method: init_method of torch.distributed.init_process_group
    :param warning: Warn if this method is called multiple times
    :return:
    """

    if not is_distributed_available():
        raise RuntimeError(
            'Distributed training is not available on this machine')

    if use_horovod:
        global IS_DISTRIBUTED_HOROVOD
        IS_DISTRIBUTED_HOROVOD = True
        if backend is not None or init_method is not None:
            raise RuntimeError(
                'Try to use horovod, but `backend` and `init_method` are not None'
            )

        if is_horovod_available():
            import horovod.torch as hvd

            hvd.init()
            loggerx.info('=> init horovod ...')
        else:
            raise RuntimeError('horovod is not available!')

    else:
        # default values
        backend = backend or 'nccl'
        init_method = init_method or 'env://'

        if not is_distributed():
            raise RuntimeError(
                f'For distributed training, use `torchrun --nproc_per_node={device_count()} COMMAND` ...')

        if distributed.is_initialized():
            loggerx.warning('`distributed` is already initialized. Skipped.')
        else:
            distributed.init_process_group(
                backend=backend, init_method=init_method)
        loggerx.info('init distributed')

    if not is_master():

        def no_print(*values, **kwargs):
            pass

        builtins.print = no_print


def is_horovod_available() -> bool:
    disable_horovod = int(os.environ.get('DISABLE_HOROVOD', 0))
    return (importlib.util.find_spec('horovod')
            is not None) and (disable_horovod == 0)


def is_distributed_available() -> bool:
    return distributed.is_available() or is_horovod_available()


def is_distributed() -> bool:
    # to handle horovod
    return get_world_size() > 1


def get_local_rank() -> int:
    # returns -1 if not distributed, else returns local rank
    # it works before dist.init_process_group
    if IS_DISTRIBUTED_HOROVOD:
        import horovod.torch as hvd

        return hvd.local_rank()
    else:
        return int(os.environ.get('LOCAL_RANK', 0))


def get_global_rank() -> int:
    # returns 0 if not distributed, else returns global rank
    # it works before dist.init_process_group
    if IS_DISTRIBUTED_HOROVOD:
        import horovod.torch as hvd

        return hvd.rank()
    else:
        return int(os.environ.get('RANK', 0))


def is_master() -> bool:
    return get_global_rank() == 0


def get_num_nodes() -> int:
    # assume all nodes have the same number of gpus
    if not is_distributed():
        return 1
    else:
        return get_world_size() // device_count()


def get_world_size() -> int:
    if IS_DISTRIBUTED_HOROVOD:
        import horovod.torch as hvd

        return hvd.size()
    else:
        return int(os.environ.get('WORLD_SIZE', 1))


def scaled_all_reduce(tensors):
    """Performs the scaled all_reduce operation on the provided tensors.

    Adapted from: https://github.com/facebookresearch/pycls/blob/master/pycls/core/distributed.py

    The input tensors are modified in-place. Currently supports only the sum
    reduction operator. The reduced values are scaled by the inverse size of the
    process group.
    """
    # There is no need for reduction in the single-proc case
    if get_world_size() == 1:
        return tensors
    # Queue the reductions
    reductions = []
    for tensor in tensors:
        reduction = distributed.all_reduce(tensor, async_op=True)
        reductions.append(reduction)
    # Wait for reductions to finish
    for reduction in reductions:
        reduction.wait()
    # Scale the results
    for tensor in tensors:
        tensor.mul_(1.0 / get_world_size())
    return tensors


def to_device(data, device=None, dtype=None, non_blocking=False, copy=False):
    """Change the device of object recursively"""
    if isinstance(data, dict):
        return {
            k: to_device(v, device, dtype, non_blocking, copy) for k, v in data.items()
        }
    elif dataclasses.is_dataclass(data) and not isinstance(data, type):
        return type(data)(
            *[
                to_device(v, device, dtype, non_blocking, copy)
                for v in dataclasses.astuple(data)
            ]
        )
    # maybe namedtuple. I don't know the correct way to judge namedtuple.
    elif isinstance(data, tuple) and type(data) is not tuple:
        return type(data)(
            *[to_device(o, device, dtype, non_blocking, copy) for o in data]
        )
    elif isinstance(data, (list, tuple)):
        return type(data)(to_device(v, device, dtype, non_blocking, copy) for v in data)
    elif isinstance(data, np.ndarray):
        return to_device(torch.from_numpy(data), device, dtype, non_blocking, copy)
    elif isinstance(data, torch.Tensor):
        return data.to(device, dtype, non_blocking, copy)
    else:
        return data


def force_gatherable(data, device):
    """Change object to gatherable in torch.nn.DataParallel recursively

    The difference from to_device() is changing to torch.Tensor if float or int
    value is found.

    The restriction to the returned value in DataParallel:
        The object must be
        - torch.cuda.Tensor
        - 1 or more dimension. 0-dimension-tensor sends warning.
        or a list, tuple, dict.

    """
    if isinstance(data, dict):
        return {k: force_gatherable(v, device) for k, v in data.items()}
    # DataParallel can't handle NamedTuple well
    elif isinstance(data, tuple) and type(data) is not tuple:
        return type(data)(*[force_gatherable(o, device) for o in data])
    elif isinstance(data, (list, tuple, set)):
        return type(data)(force_gatherable(v, device) for v in data)
    elif isinstance(data, np.ndarray):
        return force_gatherable(torch.from_numpy(data), device)
    elif isinstance(data, torch.Tensor):
        if data.dim() == 0:
            # To 1-dim array
            data = data[None]
        return data.to(device)
    elif isinstance(data, float):
        return torch.tensor([data], dtype=torch.float, device=device)
    elif isinstance(data, int):
        return torch.tensor([data], dtype=torch.long, device=device)
    elif data is None:
        return None
    else:
        loggerx.warning(f"{type(data)} may not be gatherable by DataParallel")
        return data
