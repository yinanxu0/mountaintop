from typing import Optional, Tuple
import torch


from mountaintop.layers.low_rank_transformer.low_rank_attention import LowRankMultiHeadedAttention
from mountaintop.layers.low_rank_transformer.low_rank_forward import LowRankPositionwiseFeedForward
from mountaintop.layers.base.interface import LayerInterface


class LowRankTransformerDecoderBlock(LayerInterface):
    """Single decoder layer module.

    Args:
        in_dim (int): Input dimension.
        num_heads (int): num of attention head in Self-attention module.
        num_hidden (int): num of hidden units in Feed-forward module.
        dropout_rate (float): Dropout rate.
        self_attention_dropout_rate (float): Dropout rate for Self-attention module.
        src_attention_dropout_rate (float): Dropout rate for Cross-attention module.
        norm_type (str): normalization type, should be in [prenorm, postnorm]
        concat_after (bool): Whether to concat attention layer's inpu
            and output.
            True: x -> x + linear(concat(x, att(x)))
            False: x -> x + att(x)
    """
    def __init__(
        self,
        in_dim: int,
        num_heads: int = 4,
        num_hidden: int = 2048,
        dropout_rate: float = 0.1,
        self_attention_dropout_rate: float = 0.0,
        src_attention_dropout_rate: float = 0.0,
        norm_type: str = 'prenorm',
        activation_name: str = 'relu',
        concat_after: bool = False,
        *args, **kwargs
    ):
        """Construct an DecoderLayer object."""
        super().__init__()
        assert norm_type in ["prenorm", "postnorm"]
        assert in_dim > 0
        assert num_heads > 0
        assert num_hidden > 0
        
        self.in_dim = in_dim
        self.prenorm = norm_type == "prenorm"
        self.concat_after = concat_after
        
        self.self_attn = LowRankMultiHeadedAttention(num_heads, in_dim, self_attention_dropout_rate)
        self.cross_attn = LowRankMultiHeadedAttention(num_heads, in_dim, src_attention_dropout_rate)
        self.feed_forward = LowRankPositionwiseFeedForward(in_dim, num_hidden, dropout_rate, activation_name)
        
        self.self_norm = torch.nn.LayerNorm(in_dim, eps=1e-5)
        self.cross_norm = torch.nn.LayerNorm(in_dim, eps=1e-5)
        self.feed_norm = torch.nn.LayerNorm(in_dim, eps=1e-5)
        
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.self_concat_linear = torch.nn.Linear(2*in_dim, in_dim) if concat_after else torch.nn.Identity()
        self.cross_concat_linear = torch.nn.Linear(2*in_dim, in_dim) if concat_after else torch.nn.Identity()

    def forward(
        self,
        tgt: torch.Tensor,
        tgt_mask: torch.Tensor,
        memory: torch.Tensor,
        memory_mask: torch.Tensor,
        cache: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute decoded features.

        Args:
            tgt (torch.Tensor): Input tensor (#batch, maxlen_out, in_dim).
            tgt_mask (torch.Tensor): Mask for input tensor
                (#batch, maxlen_out).
            memory (torch.Tensor): Encoded memory
                (#batch, maxlen_in, in_dim).
            memory_mask (torch.Tensor): Encoded memory mask
                (#batch, maxlen_in).
            cache (torch.Tensor): cached tensors.
                (#batch, maxlen_out - 1, in_dim).

        Returns:
            torch.Tensor: Output tensor (#batch, maxlen_out, in_dim).
            torch.Tensor: Mask for output tensor (#batch, maxlen_out).
            torch.Tensor: Encoded memory (#batch, maxlen_in, in_dim).
            torch.Tensor: Encoded memory mask (#batch, maxlen_in).

        """
        ##################
        # self attention #
        ##################
        residual = tgt
        if self.prenorm:
            tgt = self.self_norm(tgt)

        if cache is None:
            tgt_q = tgt
            tgt_q_mask = tgt_mask
        else:
            # compute only the last frame query keeping dim: max_time_out -> 1
            assert cache.shape == (tgt.shape[0], tgt.shape[1]-1, self.in_dim)
            tgt_q = tgt[:, -1:, :]
            residual = residual[:, -1:, :]
            tgt_q_mask = tgt_mask[:, -1:, :]

        if self.concat_after:
            tgt_concat = torch.cat(
                (tgt_q, self.self_attn(tgt_q, tgt, tgt, tgt_q_mask)), dim=-1)
            x = residual + self.self_concat_linear(tgt_concat)
        else:
            x = residual + self.dropout(
                self.self_attn(tgt_q, tgt, tgt, tgt_q_mask))
        
        if not self.prenorm:
            x = self.self_norm(x)

        ###################
        # cross attention #
        ###################
        residual = x
        if self.prenorm:
            x = self.cross_norm(x)
        if self.concat_after:
            x_concat = torch.cat((x, self.cross_attn(x, memory, memory, memory_mask)), dim=-1)
            x = self.cross_concat_linear(x_concat)
        else:
            x = self.dropout(self.cross_attn(x, memory, memory, memory_mask))
        x = residual + x
        
        if not self.prenorm:
            x = self.cross_norm(x)

        ################
        # feed forward #
        ################
        residual = x
        if self.prenorm:
            x = self.feed_norm(x)
        x = residual + self.feed_forward(x)
        if not self.prenorm:
            x = self.feed_norm(x)

        ##########
        # output #
        ##########
        if cache is not None:
            x = torch.cat([cache, x], dim=1)

        return x, tgt_mask, memory, memory_mask

