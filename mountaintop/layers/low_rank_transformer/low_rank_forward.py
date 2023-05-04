import torch
from typing import Optional, Any


from mountaintop.layers.activation import get_activation


class LowRankLinear(torch.nn.Module):
    def __init__(
        self, 
        in_dim: int, 
        out_dim: int, 
        inter_dim: Optional[int] = None,
        bias: bool = True,
    ) -> None:
        super().__init__()
        assert in_dim > 0, f"param `in_dim` in LowRankLinear should be positive"
        assert out_dim > 0, f"param `out_dim` in LowRankLinear should be positive"
        if inter_dim is None:
            inter_dim = max(1, ((in_dim+out_dim)//2)//8)
        assert inter_dim > 0, f"param `inter_dim` in LowRankLinear should be positive"
        
        self._linear = torch.nn.Sequential(
            torch.nn.Linear(in_dim, inter_dim, bias=bias),
            torch.nn.Linear(inter_dim, out_dim, bias=bias), 
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function.

        Args:
            x: input tensor (B, L, D)
        Returns:
            output tensor, (B, L, D)
        """
        return self._linear(x)


class LowRankPositionwiseFeedForward(torch.nn.Module):
    """Positionwise feed forward layer.

    FeedForward are appied on each position of the sequence.
    The output dim is same with the input dim.

    Args:
        in_dim (int): Input dimenstion.
        hidden_dim (int): The number of hidden units.
        dropout_rate (float): Dropout rate.
        activation (str): Activation function
    """
    def __init__(self, in_dim: int, hidden_dim: int, dropout_rate: float,
                 activation_name: str = "relu"):
        """Construct a PositionwiseFeedForward object."""
        super().__init__()
        activation_func = get_activation(activation_name)
        
        self._encoder = torch.nn.Sequential(
            LowRankLinear(in_dim, hidden_dim),
            activation_func(),
            torch.nn.Dropout(dropout_rate),
            LowRankLinear(hidden_dim, in_dim), 
            torch.nn.Dropout(dropout_rate),
        )
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function.

        Args:
            x: input tensor (B, L, D)
        Returns:
            output tensor, (B, L, D)
        """
        return self._encoder(x)
