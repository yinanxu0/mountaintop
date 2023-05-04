from typing import Union
import torch

## typing
TORCH_DP = torch.nn.DataParallel
TORCH_DDP = torch.nn.parallel.DistributedDataParallel
MODEL_TYPING = Union[torch.nn.Module, TORCH_DP, TORCH_DDP]


def extract_module_from_model(model: MODEL_TYPING):
    if isinstance(model, TORCH_DDP) or isinstance(model, TORCH_DP):
        return model.module
    else:
        return model


extract_access_model = extract_module_from_model
