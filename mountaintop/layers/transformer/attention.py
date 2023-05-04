import math
from typing import Optional
import torch


class MultiHeadedAttention(torch.nn.Module):
    """Multi-Head Attention layer.

    Args:
        num_head (int): The number of heads.
        feat_dim (int): The dimension of features.
        dropout_rate (float): Dropout rate.

    """
    def __init__(self, num_head: int, feat_dim: int, dropout_rate: float):
        """Construct an MultiHeadedAttention object."""
        super().__init__()
        assert feat_dim % num_head == 0
        # We assume d_v always equals to d_k
        self._d_k = feat_dim // num_head
        self._num_head = num_head
        self._linear_q = torch.nn.Linear(feat_dim, feat_dim)
        self._linear_k = torch.nn.Linear(feat_dim, feat_dim)
        self._linear_v = torch.nn.Linear(feat_dim, feat_dim)
        self._linear_out = torch.nn.Linear(feat_dim, feat_dim)
        self._dropout = torch.nn.Dropout(p=dropout_rate)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor], pos_emb: torch.Tensor = torch.empty(0),) -> torch.Tensor:
        """Compute scaled dot product attention.

        Args:
            query (torch.Tensor): Query tensor (#batch, time1, size).
            key (torch.Tensor): Key tensor (#batch, time2, size).
            value (torch.Tensor): Value tensor (#batch, time2, size).
            mask (torch.Tensor): Mask tensor (#batch, 1, time2) or
                (#batch, time1, time2).
                1.When applying cross attention between decoder and encoder,
                the batch padding mask for input is in (#batch, 1, T) shape.
                2.When applying self attention of encoder,
                the mask is in (#batch, T, T)  shape.
                3.When applying self attention of decoder,
                the mask is in (#batch, L, L)  shape.
                4.If the different position in decoder see different block
                of the encoder, such as Mocha, the passed in mask could be
                in (#batch, L, T) shape. But there is no such case in current
                Wenet.


        Returns:
            torch.Tensor: Output tensor (#batch, time1, d_model).

        """
        ##############################################
        # q, k, v = self.forward_qkv(query, key, value)
        n_batch = query.size(0)
        q = self._linear_q(query).view(n_batch, -1, self._num_head, self._d_k)
        k = self._linear_k(key).view(n_batch, -1, self._num_head, self._d_k)
        v = self._linear_v(value).view(n_batch, -1, self._num_head, self._d_k)
        q = q.transpose(1, 2)  # (batch, head, time1, d_k)
        k = k.transpose(1, 2)  # (batch, head, time2, d_k)
        v = v.transpose(1, 2)  # (batch, head, time2, d_k)
        ##############################################
        
        scores = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self._d_k)
        
        ##############################################
        # out = self.forward_attention(v, scores, mask)
        if mask is not None:
            mask = mask.unsqueeze(1).eq(0)  # (batch, 1, *, time2)
            scores = scores.masked_fill(mask, -float('inf'))
            attn = torch.softmax(scores, dim=-1).masked_fill(mask, 0.0)  # (batch, head, time1, time2)
        else:
            attn = torch.softmax(scores, dim=-1)  # (batch, head, time1, time2)
        p_attn = self._dropout(attn)
        x = torch.matmul(p_attn, v)  # (batch, head, time1, d_k)
        x = x.transpose(1, 2).contiguous().view(n_batch, -1, self._num_head * self._d_k)  # (batch, time1, d_model)
        out = self._linear_out(x)  # (batch, time1, d_model)
        ##############################################
        
        return out


class LegacyRelPositionMultiHeadedAttention(torch.nn.Module):
    """Multi-Head Attention layer with relative position encoding.
    Paper: https://arxiv.org/abs/1901.02860
    Args:
        num_head (int): The number of heads.
        feat_dim (int): The number of features.
        dropout_rate (float): Dropout rate.
    """
    def __init__(self, num_head: int, feat_dim: int, dropout_rate: float):
        """Construct an RelPositionMultiHeadedAttention object."""
        super().__init__()
        assert feat_dim % num_head == 0
        # We assume d_v always equals to d_k
        self._d_k = feat_dim // num_head
        self._num_head = num_head
        self._linear_q = torch.nn.Linear(feat_dim, feat_dim)
        self._linear_k = torch.nn.Linear(feat_dim, feat_dim)
        self._linear_v = torch.nn.Linear(feat_dim, feat_dim)
        self._linear_out = torch.nn.Linear(feat_dim, feat_dim)
        self._dropout = torch.nn.Dropout(p=dropout_rate)
        
        # linear transformation for positional encoding
        self._linear_pos = torch.nn.Linear(feat_dim, feat_dim, bias=False)
        # these two learnable bias are used in matrix c and matrix d
        # as described in https://arxiv.org/abs/1901.02860 Section 3.3
        self._pos_bias_u = torch.nn.Parameter(torch.Tensor(self._num_head, self._d_k))
        self._pos_bias_v = torch.nn.Parameter(torch.Tensor(self._num_head, self._d_k))
        torch.torch.nn.init.xavier_uniform_(self._pos_bias_u)
        torch.torch.nn.init.xavier_uniform_(self._pos_bias_v)

    def rel_shift(self, x, zero_triu: bool = False):
        """Compute relative positinal encoding.
        Args:
            x (torch.Tensor): Input tensor (batch, time, size).
            zero_triu (bool): If true, return the lower triangular part of
                the matrix.
        Returns:
            torch.Tensor: Output tensor.
        """
        zero_pad = torch.zeros((x.size()[0], x.size()[1], x.size()[2], 1),
                               device=x.device,
                               dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=-1)

        x_padded = x_padded.view(x.size()[0],
                                 x.size()[1],
                                 x.size(3) + 1, x.size(2))
        x = x_padded[:, :, 1:].view_as(x)

        if zero_triu:
            ones = torch.ones((x.size(2), x.size(3)))
            x = x * torch.tril(ones, x.size(3) - x.size(2))[None, None, :, :]
        return x

    def forward(self, query: torch.Tensor, key: torch.Tensor,
                value: torch.Tensor, mask: Optional[torch.Tensor],
                pos_emb: torch.Tensor):
        """Compute 'Scaled Dot Product Attention' with rel. positional encoding.
        Args:
            query (torch.Tensor): Query tensor (#batch, time1, size).
            key (torch.Tensor): Key tensor (#batch, time2, size).
            value (torch.Tensor): Value tensor (#batch, time2, size).
            mask (torch.Tensor): Mask tensor (#batch, 1, time2) or
                (#batch, time1, time2).
            pos_emb (torch.Tensor): Positional embedding tensor
                (#batch, time2, size).
        Returns:
            torch.Tensor: Output tensor (#batch, time1, d_model).
        """
        ##############################################
        # q, k, v = self.forward_qkv(query, key, value)
        n_batch = query.size(0)
        q = self._linear_q(query).view(n_batch, -1, self._num_head, self._d_k)
        k = self._linear_k(key).view(n_batch, -1, self._num_head, self._d_k)
        v = self._linear_v(value).view(n_batch, -1, self._num_head, self._d_k)
        # q : (batch, time1, head, d_k)
        k = k.transpose(1, 2)  # (batch, head, time2, d_k)
        v = v.transpose(1, 2)  # (batch, head, time2, d_k)

        n_batch_pos = pos_emb.size(0)
        p = self._linear_pos(pos_emb).view(n_batch_pos, -1, self._num_head, self._d_k)
        p = p.transpose(1, 2)  # (batch, head, time1, d_k)

        q_with_bias_u = (q + self._pos_bias_u).transpose(1, 2) # (batch, head, time1, d_k)
        q_with_bias_v = (q + self._pos_bias_v).transpose(1, 2) # (batch, head, time1, d_k)

        # compute attention score
        # first compute matrix a and matrix c
        # as described in https://arxiv.org/abs/1901.02860 Section 3.3
        # (batch, head, time1, time2)
        matrix_ac = torch.matmul(q_with_bias_u, k.transpose(2, 3))

        # compute matrix b and matrix d
        matrix_bd = torch.matmul(q_with_bias_v, p.transpose(2, 3)) # (batch, head, time1, time2)
        # Remove rel_shift since it is useless in speech recognition,
        # and it requires special attention for streaming.
        # matrix_bd = self.rel_shift(matrix_bd)

        scores = (matrix_ac + matrix_bd) / math.sqrt(self._d_k)  # (batch, head, time1, time2)

        ##############################################
        # out = self.forward_attention(v, scores, mask)
        if mask is not None:
            mask = mask.unsqueeze(1).eq(0)  # (batch, 1, *, time2)
            scores = scores.masked_fill(mask, -float('inf'))
            attn = torch.softmax(scores, dim=-1).masked_fill(mask, 0.0)  # (batch, head, time1, time2)
        else:
            attn = torch.softmax(scores, dim=-1)  # (batch, head, time1, time2)
        p_attn = self._dropout(attn)
        x = torch.matmul(p_attn, v)  # (batch, head, time1, d_k)
        x = x.transpose(1, 2).contiguous().view(n_batch, -1, self._num_head * self._d_k)  # (batch, time1, d_model)
        out = self._linear_out(x)  # (batch, time1, d_model)
        # TODO: add dropout here?
        return out


class RelPositionMultiHeadedAttention(torch.nn.Module):
    """Multi-Head Attention layer with relative position encoding.
    Paper: https://arxiv.org/abs/1901.02860
    Args:
        num_head (int): The number of heads.
        feat_dim (int): The number of features.
        dropout_rate (float): Dropout rate.
    """
    def __init__(
        self, 
        num_head: int, 
        feat_dim: int, 
        dropout_rate: float, 
        zero_triu: bool = False
    ):
        """Construct an RelPositionMultiHeadedAttention object."""
        super().__init__()
        assert feat_dim % num_head == 0
        # We assume d_v always equals to d_k
        self._d_k = feat_dim // num_head
        self._num_head = num_head
        self._zero_triu = zero_triu
        self._linear_q = torch.nn.Linear(feat_dim, feat_dim)
        self._linear_k = torch.nn.Linear(feat_dim, feat_dim)
        self._linear_v = torch.nn.Linear(feat_dim, feat_dim)
        self._linear_out = torch.nn.Linear(feat_dim, feat_dim)
        self._dropout = torch.nn.Dropout(p=dropout_rate)
        
        # linear transformation for positional encoding
        self._linear_pos = torch.nn.Linear(feat_dim, feat_dim, bias=False)
        # these two learnable bias are used in matrix c and matrix d
        # as described in https://arxiv.org/abs/1901.02860 Section 3.3
        self._pos_bias_u = torch.nn.Parameter(torch.Tensor(self._num_head, self._d_k))
        self._pos_bias_v = torch.nn.Parameter(torch.Tensor(self._num_head, self._d_k))
        torch.torch.nn.init.xavier_uniform_(self._pos_bias_u)
        torch.torch.nn.init.xavier_uniform_(self._pos_bias_v)

    def rel_shift(self, x):
        """Compute relative positinal encoding.
        Args:
            x (torch.Tensor): Input tensor (batch, time, size).
            zero_triu (bool): If true, return the lower triangular part of
                the matrix.
        Returns:
            torch.Tensor: Output tensor.
        """        
        zero_pad = torch.zeros((*x.size()[:3], 1), device=x.device, dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=-1)

        x_padded = x_padded.view(*x.size()[:2], x.size(3) + 1, x.size(2))
        x = x_padded[:, :, 1:].view_as(x)[
            :, :, :, : x.size(-1) // 2 + 1
        ]  # only keep the positions from 0 to time2

        if self._zero_triu:
            ones = torch.ones((x.size(2), x.size(3)), device=x.device)
            x = x * torch.tril(ones, x.size(3) - x.size(2))[None, None, :, :]
        return x

    def forward(self, query: torch.Tensor, key: torch.Tensor,
                value: torch.Tensor, mask: Optional[torch.Tensor],
                pos_emb: torch.Tensor):
        """Compute 'Scaled Dot Product Attention' with rel. positional encoding.
        Args:
            query (torch.Tensor): Query tensor (#batch, time1, size).
            key (torch.Tensor): Key tensor (#batch, time2, size).
            value (torch.Tensor): Value tensor (#batch, time2, size).
            mask (torch.Tensor): Mask tensor (#batch, 1, time2) or
                (#batch, time1, time2).
            pos_emb (torch.Tensor): Positional embedding tensor
                (#batch, time2, size).
        Returns:
            torch.Tensor: Output tensor (#batch, time1, d_model).
        """
        ##############################################
        # q, k, v = self.forward_qkv(query, key, value)
        batch_size = query.size(0)
        q = self._linear_q(query).view(batch_size, -1, self._num_head, self._d_k)
        k = self._linear_k(key).view(batch_size, -1, self._num_head, self._d_k)
        v = self._linear_v(value).view(batch_size, -1, self._num_head, self._d_k)
        # q : (batch, time1, head, d_k)
        k = k.transpose(1, 2)  # (batch, head, time2, d_k)
        v = v.transpose(1, 2)  # (batch, head, time2, d_k)
        
        batch_size_pos_emb = pos_emb.size(0)
        p = self._linear_pos(pos_emb).view(batch_size_pos_emb, -1, self._num_head, self._d_k)
        p = p.transpose(1, 2)  # (batch, head, time1, d_k)

        q_with_bias_u = (q + self._pos_bias_u).transpose(1, 2) # (batch, head, time1, d_k)
        q_with_bias_v = (q + self._pos_bias_v).transpose(1, 2) # (batch, head, time1, d_k)

        # compute attention score
        # first compute matrix a and matrix c
        # as described in https://arxiv.org/abs/1901.02860 Section 3.3
        # (batch, head, time1, time2)
        matrix_ac = torch.matmul(q_with_bias_u, k.transpose(2, 3))

        # compute matrix b and matrix d
        matrix_bd = torch.matmul(q_with_bias_v, p.transpose(2, 3)) # (batch, head, time1, time2)
        matrix_bd = self.rel_shift(matrix_bd)

        scores = (matrix_ac + matrix_bd) / math.sqrt(self._d_k)  # (batch, head, time1, time2)

        ##############################################
        # out = self.forward_attention(v, scores, mask)
        if mask is not None:
            mask = mask.unsqueeze(1).eq(0)  # (batch, 1, *, time2)
            scores = scores.masked_fill(mask, -float('inf'))
            attn = torch.softmax(scores, dim=-1).masked_fill(mask, 0.0)  # (batch, head, time1, time2)
        else:
            attn = torch.softmax(scores, dim=-1)  # (batch, head, time1, time2)
        p_attn = self._dropout(attn)
        x = torch.matmul(p_attn, v)  # (batch, head, time1, d_k)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self._num_head * self._d_k)  # (batch, time1, d_model)
        out = self._linear_out(x)  # (batch, time1, d_model)
        # TODO: add dropout here?
        return out
