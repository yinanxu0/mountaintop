from typing import Optional, Tuple
import torch
from torch import nn


from mountaintop.layers.activation import get_activation


class ConformerConv(nn.Module):
    """ConformerConv in Conformer model."""
    def __init__(self,
                 channels: int,
                 kernel_size: int = 15,
                 activation_name: str = "swish",
                 norm_type: str = "batchnorm",
                 causal: bool = False,
                 bias: bool = True):
        """Construct an ConformerConv object.
        Args:
            channels (int): The number of channels of conv layers.
            kernel_size (int): Kernel size of conv layers.
            causal (int): Whether use causal convolution or not
        """
        # assert check_argument_types()
        super().__init__()
        assert norm_type in ["batchnorm", "layernorm"]

        self._pointwise_conv1 = nn.Conv1d(channels, 2*channels, kernel_size=1, 
                                         stride=1, padding=0, bias=bias)
        # self.lorder is used to distinguish if it's a causal convolution,
        # if self.lorder > 0: it's a causal convolution, the input will be
        #    padded with self.lorder frames on the left in forward.
        # else: it's a symmetrical convolution
        if causal:
            padding = 0
            self.lorder = kernel_size - 1
        else:
            # kernel_size should be an odd number for none causal convolution
            assert (kernel_size - 1) % 2 == 0
            padding = (kernel_size - 1) // 2
            self.lorder = 0
        self._depthwise_conv = nn.Conv1d(
            in_channels=channels, out_channels=channels, kernel_size=kernel_size,
            stride=1, padding=padding, groups=channels, bias=bias,
        )

        self.use_layer_norm = (norm_type == "layernorm")
        self._norm = nn.BatchNorm1d(channels) if norm_type == "batchnorm" else nn.LayerNorm(channels)

        self._pointwise_conv2 = nn.Conv1d(
            in_channels=channels, out_channels=channels, kernel_size=1,
            stride=1, padding=0, bias=bias,
        )
        self.activation = get_activation(activation_name)()

    def forward( self, x: torch.Tensor, mask_pad: Optional[torch.Tensor] = None, cache: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute convolution module.
        Args:
            x (torch.Tensor): Input tensor (#batch, time, channels).
            mask_pad (torch.Tensor): used for batch padding (#batch, 1, time)
            cache (torch.Tensor): left context cache, it is only
                used in causal convolution
        Returns:
            torch.Tensor: Output tensor (#batch, time, channels).
        """
        # exchange the temporal dimension and the feature dimension
        x = x.transpose(1, 2)  # (#batch, channels, time)
        
        # mask batch padding
        if mask_pad is not None:
            x = mask_pad * x

        if self.lorder > 0:
            if cache is None:
                x = nn.functional.pad(x, (self.lorder, 0), 'constant', 0.0)
            else:
                assert cache.size(0) == x.size(0)
                assert cache.size(1) == x.size(1)
                x = torch.cat((cache, x), dim=2)
            assert (x.size(2) > self.lorder)
            new_cache = x[:, :, -self.lorder:]
        else:
            # It's better we just return None if no cache is requried,
            # However, for JIT export, here we just fake one tensor instead of
            # None.
            new_cache = torch.tensor([0.0], dtype=x.dtype, device=x.device)

        # GLU mechanism
        x = self._pointwise_conv1(x)  # (batch, 2*channel, dim)
        x = nn.functional.glu(x, dim=1)  # (batch, channel, dim)

        # 1D Depthwise Conv
        x = self._depthwise_conv(x)
        if self.use_layer_norm:
            x = x.transpose(1, 2)
        x = self.activation(self._norm(x))
        if self.use_layer_norm:
            x = x.transpose(1, 2)
        x = self._pointwise_conv2(x)
        # mask batch padding
        if mask_pad is not None:
            x = mask_pad * x
            # x.masked_fill_(~mask_pad, 0.0)

        return x.transpose(1, 2), new_cache
