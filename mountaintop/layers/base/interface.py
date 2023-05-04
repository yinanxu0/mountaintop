import torch



from mountaintop.layers.base.dataclass import (
    LayerResult,
    EncoderResult,
    EncoderBlockResult,
    DecoderResult,
    DecoderBlockResult
    
)

#### Base Layer Interface ####
class LayerInterface(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, *args, **kwargs) -> LayerResult:
        error_msg = f"forward method not implemented in {self.__class__}"
        if len(args) > 0:
            error_msg += f"\n\targs: {args}"
        if len(kwargs) > 0:
            error_msg += f"\n\tkwargs: {kwargs}"
        raise NotImplementedError(error_msg)


#### Specific Layer Interface ####
class BaseEncoderBlockInterface(LayerInterface):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, *args, **kwargs) -> EncoderBlockResult:
        super().forward(*args, **kwargs)


class BaseEncoderInterface(LayerInterface):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, *args, **kwargs) -> EncoderResult:
        super().forward(*args, **kwargs)


class BaseDecoderBlockInterface(LayerInterface):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, *args, **kwargs) -> DecoderBlockResult:
        super().forward(*args, **kwargs)


class BaseDecoderInterface(LayerInterface):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, *args, **kwargs) -> DecoderResult:
        super().forward(*args, **kwargs)






