import torch


from mountaintop.layers.activation import get_activation


class PositionwiseFeedForward(torch.nn.Module):
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
            torch.nn.Linear(in_dim, hidden_dim),
            activation_func(),
            torch.nn.Dropout(dropout_rate),
            torch.nn.Linear(hidden_dim, in_dim), 
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
