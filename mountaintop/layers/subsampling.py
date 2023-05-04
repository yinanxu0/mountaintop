from typing import Tuple
import torch


class LinearNoSubsampling(torch.nn.Module):
    """Linear transform the input without subsampling

    Args:
        in_dim (int): Input dimension.
        out_dim (int): Output dimension.
        dropout_rate (float): Dropout rate.

    """
    def __init__(self, in_dim: int, out_dim: int, dropout_rate: float):
        """Construct an linear object."""
        super().__init__()
        self._encoder = torch.nn.Sequential(
            torch.nn.Linear(in_dim, out_dim),
            torch.nn.LayerNorm(out_dim, eps=1e-5),
            torch.nn.Dropout(dropout_rate),
        )
        # self._pos_enc = pos_enc
        ## using default
        self._num_right_context = 0
        self._subsampling_rate = 1

    def forward(self, x: torch.Tensor, mask: torch.Tensor, offset: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        """Input x.

        Args:
            x (torch.Tensor): Input tensor (#batch, time, in_dim).
            mask (torch.Tensor): Input mask (#batch, 1, time).

        Returns:
            torch.Tensor: linear input tensor (#batch, time', out_dim),
                where time' = time .
            torch.Tensor: linear input mask (#batch, 1, time'),
                where time' = time .

        """
        x = self._encoder(x)
        return x, mask
    
    @property
    def rate(self):
        return self._subsampling_rate

    @property
    def right_context(self):
        return self._num_right_context
    

class Conv2dSubsampling4(torch.nn.Module):
    """Convolutional 2D subsampling (to 1/4 length).

    Args:
        in_dim (int): Input dimension.
        out_dim (int): Output dimension.
        dropout_rate (float): Dropout rate.

    """
    def __init__(self, in_dim: int, out_dim: int, dropout_rate: float):
        """Construct an Conv2dSubsampling4 object."""
        assert in_dim >= 7
        super().__init__()
        self._encoder = torch.nn.Sequential(
            torch.nn.Conv2d(1, out_dim, 3, 2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(out_dim, out_dim, 3, 2),
            torch.nn.ReLU(),
        )
        self._linear = torch.nn.Linear(out_dim * (((in_dim - 1) // 2 - 1) // 2), out_dim)
        # self._pos_enc = pos_enc
        # The right context for every conv layer is computed by:
        # (kernel_size - 1) / 2 * stride  * frame_rate_of_this_layer
        self._subsampling_rate = 4
        # 6 = (3 - 1) / 2 * 2 * 1 + (3 - 1) / 2 * 2 * 2
        self._num_right_context = 6

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor, offset: int = 0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Subsample x.

        Args:
            x (torch.Tensor): Input tensor (#batch, time, in_dim).
            mask (torch.Tensor): Input mask (#batch, 1, time).

        Returns:
            torch.Tensor: Subsampled tensor (#batch, time', out_dim),
                where time' = time // 4.
            torch.Tensor: Subsampled mask (#batch, 1, time'),
                where time' = time // 4.
            torch.Tensor: positional encoding

        """
        x = x.unsqueeze(1)  # (b, c, t, f)
        x = self._encoder(x)
        b, c, t, f = x.size()
        x = self._linear(x.transpose(1, 2).contiguous().view(b, t, c*f))
        return x, mask[:, :, :-2:2][:, :, :-2:2]

    @property
    def rate(self):
        return self._subsampling_rate
    
    @property
    def right_context(self):
        return self._num_right_context
    

class Conv2dSubsampling6(torch.nn.Module):
    """Convolutional 2D subsampling (to 1/6 length).
    Args:
        in_dim (int): Input dimension.
        out_dim (int): Output dimension.
        dropout_rate (float): Dropout rate.
        pos_enc (torch.nn.Module): Custom position encoding layer.
    """
    def __init__(self, in_dim: int, out_dim: int, dropout_rate: float):
        """Construct an Conv2dSubsampling6 object."""
        super().__init__()
        self._encoder = torch.nn.Sequential(
            torch.nn.Conv2d(1, out_dim, 3, 2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(out_dim, out_dim, 5, 3),
            torch.nn.ReLU(),
        )
        self._linear = torch.nn.Linear(out_dim * (((in_dim - 1) // 2 - 2) // 3), out_dim)
        # self._pos_enc = pos_enc
        # 14 = (3 - 1) / 2 * 2 * 1 + (5 - 1) / 2 * 3 * 2
        self._subsampling_rate = 6
        self._num_right_context = 14

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor, offset: int = 0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Subsample x.
        Args:
            x (torch.Tensor): Input tensor (#batch, time, in_dim).
            mask (torch.Tensor): Input mask (#batch, 1, time).

        Returns:
            torch.Tensor: Subsampled tensor (#batch, time', out_dim),
                where time' = time // 6.
            torch.Tensor: Subsampled mask (#batch, 1, time'),
                where time' = time // 6.
            torch.Tensor: positional encoding
        """
        x = x.unsqueeze(1)  # (b, c, t, f)
        x = self._encoder(x)
        b, c, t, f = x.size()
        x = self._linear(x.transpose(1, 2).contiguous().view(b, t, c*f))
        return x, mask[:, :, :-2:2][:, :, :-4:3]

    @property
    def rate(self):
        return self._subsampling_rate
    
    @property
    def right_context(self):
        return self._num_right_context
    

class Conv2dSubsampling8(torch.nn.Module):
    """Convolutional 2D subsampling (to 1/8 length).

    Args:
        in_dim (int): Input dimension.
        out_dim (int): Output dimension.
        dropout_rate (float): Dropout rate.

    """
    def __init__(self, in_dim: int, out_dim: int, dropout_rate: float):
        """Construct an Conv2dSubsampling8 object."""
        super().__init__()
        self._encoder = torch.nn.Sequential(
            torch.nn.Conv2d(1, out_dim, 3, 2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(out_dim, out_dim, 3, 2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(out_dim, out_dim, 3, 2),
            torch.nn.ReLU(),
        )
        self._linear = torch.nn.Linear(
            out_dim * ((((in_dim - 1) // 2 - 1) // 2 - 1) // 2), out_dim)
        # self._pos_enc = pos_enc
        self._subsampling_rate = 8
        # 14 = (3 - 1) / 2 * 2 * 1 + (3 - 1) / 2 * 2 * 2 + (3 - 1) / 2 * 2 * 4
        self._num_right_context = 14

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor, offset: int = 0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Subsample x.

        Args:
            x (torch.Tensor): Input tensor (#batch, time, in_dim).
            mask (torch.Tensor): Input mask (#batch, 1, time).

        Returns:
            torch.Tensor: Subsampled tensor (#batch, time', out_dim),
                where time' = time // 8.
            torch.Tensor: Subsampled mask (#batch, 1, time'),
                where time' = time // 8.
            torch.Tensor: positional encoding
        """
        x = x.unsqueeze(1)  # (b, c, t, f)
        x = self._encoder(x)
        b, c, t, f = x.size()
        x = self._linear(x.transpose(1, 2).contiguous().view(b, t, c*f))
        return x, mask[:, :, :-2:2][:, :, :-2:2][:, :, :-2:2]

    @property
    def rate(self):
        return self._subsampling_rate
    
    @property
    def right_context(self):
        return self._num_right_context

  