# from collections import namedtuple
import torch
from typing import List, NamedTuple



LayerResult = NamedTuple


###################
# Encoder Results #
###################
# encoder block
EncoderBlockResult = NamedTuple(
    'EncoderBlockResult', 
    output=torch.Tensor,
    mask=torch.Tensor,
    attention_cache=torch.Tensor,
    cnn_cache=torch.Tensor
)

# encoder
EncoderResult = NamedTuple(
    'EncoderResult', 
    output=torch.Tensor,
    mask=torch.Tensor,
    inner_output=torch.Tensor,
)


###################
# Decoder Results #
###################
# decoder block
DecoderBlockResult = NamedTuple(
    'DecoderBlockResult', 
    output=torch.Tensor,
    mask=torch.Tensor,
    attention_cache=torch.Tensor,
    cnn_cache=torch.Tensor
)

# decoder
DecoderResult = NamedTuple(
    'DecoderResult', 
    output=torch.Tensor,
    mask=torch.Tensor,
    inner_output=List[torch.Tensor],
)



