import math
from typing import Tuple
import torch


class PositionalEncoding(torch.nn.Module):
    """Positional encoding.

    :param int dim: embedding dim
    :param float dropout_rate: dropout rate
    :param int max_len: maximum input length

    PE(pos, 2i)   = sin(pos/(10000^(2i/dmodel)))
    PE(pos, 2i+1) = cos(pos/(10000^(2i/dmodel)))
    """
    def __init__(self, dim: int, dropout_rate: float, max_len: int = 5000, *args, **kwargs):
        """Construct an PositionalEncoding object."""
        super().__init__()
        self._dim = dim
        self._xscale = math.sqrt(self._dim)
        self._dropout = torch.nn.Dropout(p=dropout_rate)
        self._max_len = max_len

        self._pe = torch.zeros(size=(self._max_len, self._dim))
        position = torch.arange(0, self._max_len,
                                dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self._dim, 2, dtype=torch.float32) *
            -(math.log(10000.0) / self._dim))
        self._pe[:, 0::2] = torch.sin(position * div_term)
        self._pe[:, 1::2] = torch.cos(position * div_term)
        self._pe = self._pe.unsqueeze(0)

    def forward(self, x: torch.Tensor, offset: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        """Add positional encoding.

        Args:
            x (torch.Tensor): Input. Its shape is (batch, time, ...)
            offset (int): position offset

        Returns:
            torch.Tensor: Encoded tensor. Its shape is (batch, time, ...)
            torch.Tensor: for compatibility to RelPositionalEncoding
        """
        assert offset + x.size(1) < self._max_len
        self._pe = self._pe.to(x.device)
        pos_emb = self._pe[:, offset:offset + x.size(1)]
        x = x * self._xscale + pos_emb
        return self._dropout(x), self._dropout(pos_emb)

    def position_encoding(self, offset: int, size: int, *args, **kwargs) -> torch.Tensor:
        """ For getting encoding in a streaming fashion

        Attention!!!!!
        we apply dropout only once at the whole utterance level in a none
        streaming way, but will call this function several times with
        increasing input size in a streaming scenario, so the dropout will
        be applied several times.

        Args:
            offset (int): start offset
            size (int): requried size of position encoding

        Returns:
            torch.Tensor: Corresponding encoding
        """
        assert offset + size < self._max_len
        return self._dropout(self._pe[:, offset:offset+size])


class NoScalePositionalEncoding(torch.nn.Module):
    """Positional encoding.

    :param int dim: embedding dim
    :param float dropout_rate: dropout rate
    :param int max_len: maximum input length

    PE(pos, 2i)   = sin(pos/(10000^(2i/dmodel)))
    PE(pos, 2i+1) = cos(pos/(10000^(2i/dmodel)))
    """
    def __init__(self, dim: int, dropout_rate: float, max_len: int = 5000, *args, **kwargs):
        """Construct an PositionalEncoding object."""
        super().__init__()
        self._dim = dim
        self._xscale = 1.0
        self._dropout = torch.nn.Dropout(p=dropout_rate)
        self._max_len = max_len

        self._pe = torch.zeros(size=(self._max_len, self._dim))
        position = torch.arange(0, self._max_len,
                                dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self._dim, 2, dtype=torch.float32) *
            -(math.log(10000.0) / self._dim))
        self._pe[:, 0::2] = torch.sin(position * div_term)
        self._pe[:, 1::2] = torch.cos(position * div_term)
        self._pe = self._pe.unsqueeze(0)

    def forward(self, x: torch.Tensor, offset: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        """Add positional encoding.

        Args:
            x (torch.Tensor): Input. Its shape is (batch, time, ...)
            offset (int): position offset

        Returns:
            torch.Tensor: Encoded tensor. Its shape is (batch, time, ...)
            torch.Tensor: for compatibility to RelPositionalEncoding
        """
        assert offset + x.size(1) < self._max_len
        self._pe = self._pe.to(x.device)
        pos_emb = self._pe[:, offset:offset + x.size(1)]
        x = x * self._xscale + pos_emb
        return self._dropout(x), self._dropout(pos_emb)

    def position_encoding(self, offset: int, size: int, *args, **kwargs) -> torch.Tensor:
        """ For getting encoding in a streaming fashion

        Attention!!!!!
        we apply dropout only once at the whole utterance level in a none
        streaming way, but will call this function several times with
        increasing input size in a streaming scenario, so the dropout will
        be applied several times.

        Args:
            offset (int): start offset
            size (int): requried size of position encoding

        Returns:
            torch.Tensor: Corresponding encoding
        """
        assert offset + size < self._max_len
        return self._dropout(self._pe[:, offset:offset+size])


class RelPositionalEncoding(torch.nn.Module):
    """Relative positional encoding module.

    Details can be found in https://github.com/espnet/espnet/pull/2816.

    See : Appendix B in https://arxiv.org/abs/1901.02860

    Args:
        dim (int): Embedding dimension.
        dropout_rate (float): Dropout rate.
        max_len (int): Maximum input length.

    """
    def __init__(self, dim: int, dropout_rate: float, max_len: int = 5000, *args, **kwargs):
        """Construct an PositionalEncoding object."""
        super().__init__()
        self._dim = dim
        self._max_len = max_len
        self._xscale = math.sqrt(self._dim)
        self._dropout = torch.nn.Dropout(p=dropout_rate)
        self._pe = None
        self.extend_pe(max_len)

    def extend_pe(self, length):
        """Reset the positional encodings."""
        if self._pe is not None and self._pe.size(1) >= length * 2 - 1:
            return
        # Suppose `i` means to the position of query vecotr and `j` means the
        # position of key vector. We use position relative positions when keys
        # are to the left (i>j) and negative relative positions otherwise (i<j).
        pe_positive = torch.zeros(length, self._dim)
        pe_negative = torch.zeros(length, self._dim)
        position = torch.arange(0, length, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self._dim, 2, dtype=torch.float32)
            * -(math.log(10000.0) / self._dim)
        )
        pe_positive[:, 0::2] = torch.sin(position * div_term)
        pe_positive[:, 1::2] = torch.cos(position * div_term)
        pe_negative[:, 0::2] = torch.sin(-1 * position * div_term)
        pe_negative[:, 1::2] = torch.cos(-1 * position * div_term)

        # Reserve the order of positive indices and concat both positive and
        # negative indices. This is used to support the shifting trick
        # as in https://arxiv.org/abs/1901.02860
        pe_positive = torch.flip(pe_positive, [0]).unsqueeze(0)
        pe_negative = pe_negative[1:].unsqueeze(0)
        self._pe = torch.cat([pe_positive, pe_negative], dim=1)

    def forward(
            self, x: torch.Tensor, offset: int = 0
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Add positional encoding.

        Args:
            x (torch.Tensor): Input tensor (batch, time, `*`).

        Returns:
            torch.Tensor: Encoded tensor (batch, time, `*`).

        """
        self.extend_pe(offset+x.size(1))
        self._pe = self._pe.to(x.device)
        x = x * self._xscale
        pos_emb = self._pe[
            :,
            offset + self._pe.size(1) // 2 - x.size(1) + 1 : offset + self._pe.size(1) // 2 + x.size(1),
        ]
        return self._dropout(x), self._dropout(pos_emb)

    def position_encoding(self, offset: int, size: int, *args, **kwargs) -> torch.Tensor:
        """ For getting encoding in a streaming fashion

        Attention!!!!!
        we apply dropout only once at the whole utterance level in a none
        streaming way, but will call this function several times with
        increasing input size in a streaming scenario, so the dropout will
        be applied several times.

        Args:
            offset (int): start offset
            size (int): requried size of position encoding

        Returns:
            torch.Tensor: Corresponding encoding
        """
        self.extend_pe(offset+size)
        pos_emb = self._pe[
            :,
            offset + self._pe.size(1) // 2 - size + 1 : offset + self._pe.size(1) // 2 + size,
        ]
        return self._dropout(pos_emb)
    

class NoScaleRelPositionalEncoding(torch.nn.Module):
    """Relative positional encoding module.

    Details can be found in https://github.com/espnet/espnet/pull/2816.

    See : Appendix B in https://arxiv.org/abs/1901.02860

    Args:
        dim (int): Embedding dimension.
        dropout_rate (float): Dropout rate.
        max_len (int): Maximum input length.

    """
    def __init__(self, dim: int, dropout_rate: float, max_len: int = 5000, *args, **kwargs):
        """Construct an PositionalEncoding object."""
        super().__init__()
        self._dim = dim
        self._max_len = max_len
        self._xscale = 1.0
        self._dropout = torch.nn.Dropout(p=dropout_rate)
        self._pe = None
        self.extend_pe(max_len)

    def extend_pe(self, length):
        """Reset the positional encodings."""
        if self._pe is not None and self._pe.size(1) >= length * 2 - 1:
            return
        # Suppose `i` means to the position of query vecotr and `j` means the
        # position of key vector. We use position relative positions when keys
        # are to the left (i>j) and negative relative positions otherwise (i<j).
        pe_positive = torch.zeros(length, self._dim)
        pe_negative = torch.zeros(length, self._dim)
        position = torch.arange(0, length, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self._dim, 2, dtype=torch.float32)
            * -(math.log(10000.0) / self._dim)
        )
        pe_positive[:, 0::2] = torch.sin(position * div_term)
        pe_positive[:, 1::2] = torch.cos(position * div_term)
        pe_negative[:, 0::2] = torch.sin(-1 * position * div_term)
        pe_negative[:, 1::2] = torch.cos(-1 * position * div_term)

        # Reserve the order of positive indices and concat both positive and
        # negative indices. This is used to support the shifting trick
        # as in https://arxiv.org/abs/1901.02860
        pe_positive = torch.flip(pe_positive, [0]).unsqueeze(0)
        pe_negative = pe_negative[1:].unsqueeze(0)
        self._pe = torch.cat([pe_positive, pe_negative], dim=1)

    def forward(
            self, x: torch.Tensor, offset: int = 0
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Add positional encoding.

        Args:
            x (torch.Tensor): Input tensor (batch, time, `*`).

        Returns:
            torch.Tensor: Encoded tensor (batch, time, `*`).

        """
        self.extend_pe(offset+x.size(1))
        self._pe = self._pe.to(x.device)
        x = x * self._xscale
        pos_emb = self._pe[
            :,
            offset + self._pe.size(1) // 2 - x.size(1) + 1 : offset + self._pe.size(1) // 2 + x.size(1),
        ]
        return self._dropout(x), self._dropout(pos_emb)

    def position_encoding(self, offset: int, size: int, *args, **kwargs) -> torch.Tensor:
        """ For getting encoding in a streaming fashion

        Attention!!!!!
        we apply dropout only once at the whole utterance level in a none
        streaming way, but will call this function several times with
        increasing input size in a streaming scenario, so the dropout will
        be applied several times.

        Args:
            offset (int): start offset
            size (int): requried size of position encoding

        Returns:
            torch.Tensor: Corresponding encoding
        """
        self.extend_pe(offset+size)
        pos_emb = self._pe[
            :,
            offset + self._pe.size(1) // 2 - size + 1 : offset + self._pe.size(1) // 2 + size,
        ]
        return self._dropout(pos_emb)


class NoPositionalEncoding(torch.nn.Module):
    """ No position encoding
    """
    def __init__(self, dim: int, dropout_rate: float, *args, **kwargs):
        super().__init__()
        self._dim = dim
        self._dropout = torch.nn.Dropout(p=dropout_rate)

    def forward(self, x: torch.Tensor, offset: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Just return zero vector for interface compatibility
        """
        pos_emb = torch.zeros(1, x.size(1), self._dim).to(x.device)
        return self._dropout(x), pos_emb

    def position_encoding(self, size: int, *args, **kwargs) -> torch.Tensor:
        return torch.zeros(1, size, self._dim)
