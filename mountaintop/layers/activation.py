import torch
from typing import Type


from mountaintop.runx.logx import loggerx


ACTIVATION_MAP = {
    "hardtanh": torch.nn.Hardtanh,
    "tanh": torch.nn.Tanh,
    "relu": torch.nn.ReLU,
    "selu": torch.nn.SELU,
    "gelu": torch.nn.GELU
}


def register_activation(cls):
    cls_name = cls.__name__.lower()
    ACTIVATION_MAP[cls_name] = cls
    return cls


@register_activation
class Swish(torch.nn.Module):
    """Construct an Swish object."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return Swish activation function."""
        return x * torch.sigmoid(x)


class DoubleSwishFunction(torch.autograd.Function):
    """
      double_swish(x) = x * torch.sigmoid(x-1)
    This is a definition, originally motivated by its close numerical
    similarity to swish(swish(x)), where swish(x) =  x * sigmoid(x).

    Memory-efficient derivative computation:
     double_swish(x) = x * s, where s(x) = torch.sigmoid(x-1)
     double_swish'(x) = d/dx double_swish(x) =  x * s'(x) + x' * s(x) = x * s'(x) + s(x).
     Now, s'(x) = s(x) * (1-s(x)).
     double_swish'(x) =  x * s'(x) + s(x).
                      =  x * s(x) * (1-s(x)) + s(x).
                     = double_swish(x) * (1-s(x)) + s(x)
     ... so we just need to remember s(x) but not x itself.
    """
    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        x = x.detach()
        s = torch.sigmoid(x - 1.0)
        y = x * s
        ctx.save_for_backward(s, y)
        return y

    @staticmethod
    def backward(ctx, y_grad: torch.Tensor) -> torch.Tensor:
        s, y = ctx.saved_tensors
        return (y * (1 - s) + s) * y_grad


@register_activation
class DoubleSwish(torch.nn.Module):
    """Construct a DoubleSwish object."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return double-swish activation function which is an approximation to Swish(Swish(x)),
        that we approximate closely with x * sigmoid(x-1).
        """
        if torch.jit.is_scripting():
            return x * torch.sigmoid(x - 1.0)
        else:
            return DoubleSwishFunction.apply(x)


def get_activation(name: str) -> Type[torch.nn.Module]:
    """Return activation function."""
    name = name.lower()
    if name not in ACTIVATION_MAP:
        loggerx.warning(
            f"{name} is not a valid activation. activation should be in "
            f"{ACTIVATION_MAP.keys()}, Using default activation ReLU")
        name = "relu"
    return ACTIVATION_MAP[name]

