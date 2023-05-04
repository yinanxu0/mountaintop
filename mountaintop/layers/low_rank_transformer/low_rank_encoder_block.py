from typing import Optional, Tuple
import torch


from mountaintop.layers.low_rank_transformer.low_rank_attention import (
    LowRankMultiHeadedAttention, 
    LowRankRelPositionMultiHeadedAttention
)
from mountaintop.layers.low_rank_transformer.low_rank_forward import LowRankPositionwiseFeedForward
from mountaintop.layers.transformer.convolution import ConformerConv
from mountaintop.core.internal.module import import_module
from mountaintop.layers.base.interface import LayerInterface


class LowRankTransformerEncoderBlock(LayerInterface):
    """Encoder layer module.

    Args:
        dim (int): Input dimension.
        num_heads (int): num of attention head in Self-attention module.
        num_hidden (int): num of hidden units in Feed-forward module.
        dropout_rate (float): Dropout rate.
        attention_dropout_rate (float): Dropout rate for Self-attention module.
        norm_type (str): normalization type, should be in [prenorm, postnorm]
        ##############################################
        concat_after (bool): Whether to concat attention layer's input and
            output.
            True: x -> x + linear(concat(x, att(x)))
            False: x -> x + att(x)
    """
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_hidden: int,
        dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.0,
        norm_type: str = 'prenorm',
        activation_name: str = 'relu',
        concat_after: bool = False,
        *args, **kwargs
    ):
        """Construct an YXTransformerEncoderBlock object."""
        super().__init__()
        # check args
        assert norm_type in ["prenorm", "postnorm"]
        assert dim > 0
        assert num_heads > 0
        assert num_hidden > 0
        
        self.dim = dim
        if "attention_module" in kwargs:
            Attention_Block = import_module(kwargs["attention_module"])
        else:
            Attention_Block = LowRankMultiHeadedAttention 
        self.self_attn = Attention_Block(num_heads, dim, attention_dropout_rate)
        self.feed_forward = LowRankPositionwiseFeedForward(dim, num_hidden, dropout_rate, activation_name)
        self.attn_norm = torch.nn.LayerNorm(dim, eps=1e-5)
        self.feed_norm = torch.nn.LayerNorm(dim, eps=1e-5)
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.prenorm = norm_type == "prenorm"
        # concat_linear may be not used in forward fuction,
        # but will be saved in the *.pt
        self.concat_after = concat_after
        self.concat_linear = torch.nn.Linear(2*dim, dim) if concat_after else torch.nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        pos_emb: torch.Tensor,
        mask_pad: Optional[torch.Tensor] = torch.ones((0, 0, 0), dtype=torch.bool),
        attn_cache: Optional[torch.Tensor] = None,
        cnn_cache: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute encoded features.

        Args:
            x (torch.Tensor): Input tensor (#batch, time, size).
            mask (torch.Tensor): Mask tensor for the input (#batch, time).
            pos_emb (torch.Tensor): just for interface compatibility
                to ConformerEncoderLayer
            mask_pad (torch.Tensor): does not used in transformer layer,
                just for unified api with conformer.
            attn_cache (torch.Tensor): Cache tensor of the output
                (#batch, time2, size), time2 < time in x.
            cnn_cache (torch.Tensor): not used here, it's for interface
                compatibility to ConformerEncoderLayer
        Returns:
            torch.Tensor: Output tensor (#batch, time, size).
            torch.Tensor: Mask tensor (#batch, time).

        """
        ##################
        ## self attention
        ##################
        residual = x
        if self.prenorm:
            x = self.attn_norm(x)
        
        attn_cache_is_none = (attn_cache is None or attn_cache.size(0) == 0)

        if attn_cache_is_none:
            x_q = x
        else:
            assert attn_cache.size(0) == x.size(0)
            assert attn_cache.size(1) < x.size(1)
            assert attn_cache.size(2) == self.dim
            offset = attn_cache.size(1)
            x_q = x[:, offset:, :]
            residual = residual[:, offset:, :]
            mask = mask[:, offset:, :]
        
        x_attn = self.self_attn(x_q, x, x, mask, pos_emb)
        if self.concat_after:
            x_concat = torch.cat((x, x_attn), dim=-1)
            x = residual + self.concat_linear(x_concat)
        else:
            x = residual + self.dropout(x_attn)
        if not self.prenorm:
            x = self.attn_norm(x)
        
        ################
        ## feed-forward
        ################
        residual = x
        if self.prenorm:
            x = self.feed_norm(x)

        x = residual + self.feed_forward(x)
        
        if not self.prenorm:
            x = self.feed_norm(x)

        if not attn_cache_is_none:
            x = torch.cat([attn_cache, x], dim=1)
        
        new_attn_cache = x
        fake_cnn_cache = torch.tensor([0.0], dtype=x.dtype, device=x.device)
        return x, mask, new_attn_cache, fake_cnn_cache


class LowRankConformerEncoderBlock(LayerInterface):
    """Encoder layer module.
    Args:
        size (int): Input dimension.
        self_attn (torch.nn.Module): Self-attention module instance.
            `LowRankMultiHeadedAttention` or `LowRankRelPositionMultiHeadedAttention`
            instance can be used as the argument.
        feed_forward (torch.nn.Module): Feed-forward module instance.
            `LowRankPositionwiseFeedForward` instance can be used as the argument.
        feed_forward_macaron (torch.nn.Module): Additional feed-forward module
             instance.
            `LowRankPositionwiseFeedForward` instance can be used as the argument.
        conv_module (torch.nn.Module): Convolution module instance.
            `ConvlutionModule` instance can be used as the argument.
        dropout_rate (float): Dropout rate.
        normalize_before (bool):
            True: use layer_norm before each sub-block.
            False: use layer_norm after each sub-block.
        concat_after (bool): Whether to concat attention layer's input and
            output.
            True: x -> x + linear(concat(x, att(x)))
            False: x -> x + att(x)
    """
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_hidden: int,
        dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.0,
        norm_type: str = "prenorm",
        activation_name: str = "relu", # TODO: feed-forward to swish???
        concat_after: bool = False,
        use_macaron: bool = True,
        use_cnn: bool = True,
        cnn_kernels: int = 15,
        cnn_norm: str = "batchnorm", 
        cnn_causal: bool = True,
        cnn_activation_name: str = "swish",
        *args, **kwargs
    ):
        """Construct an EncoderLayer object."""
        super().__init__()
        # check args
        assert norm_type in ["prenorm", "postnorm"]
        assert cnn_norm in ["batchnorm", "layernorm"]
        assert dim > 0
        assert num_heads > 0
        assert num_hidden > 0

        self.dim = dim
        if "attention_module" in kwargs:
            Attention_Block = import_module(kwargs["attention_module"])
        else:
            Attention_Block = LowRankRelPositionMultiHeadedAttention 
        self.self_attn = Attention_Block(num_heads, dim, attention_dropout_rate)
        self.feed_forward = LowRankPositionwiseFeedForward(dim, num_hidden, dropout_rate, activation_name)
        self.attn_norm = torch.nn.LayerNorm(dim, eps=1e-5)
        self.feed_norm = torch.nn.LayerNorm(dim, eps=1e-5)
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.prenorm = norm_type == "prenorm"
        # concat_linear may be not used in forward fuction,
        # but will be saved in the *.pt
        self.concat_after = concat_after
        self.concat_linear = torch.nn.Linear(2*dim, dim) if concat_after else None
        
        ## conformer specific part
        self.macaron_feed_forward = LowRankPositionwiseFeedForward(dim, num_hidden, dropout_rate, 
            activation_name) if use_macaron else None
        self.macaron_norm = torch.nn.LayerNorm(dim, eps=1e-5) if use_macaron else None
        self.macaron_scale = 0.5 if use_macaron else 1.0
        
        self.conv_module = ConformerConv(
            channels=dim, 
            kernel_size=cnn_kernels, 
            activation_name=cnn_activation_name, 
            norm_type=cnn_norm, 
            causal=cnn_causal
        ) if use_cnn else None
        self.conv_norm = torch.nn.LayerNorm(dim, eps=1e-5) if use_cnn else None
        self.final_norm = torch.nn.LayerNorm(dim, eps=1e-5) if use_cnn else None
        
    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        pos_emb: torch.Tensor,
        mask_pad: Optional[torch.Tensor] = None,
        attn_cache: Optional[torch.Tensor] = None,
        cnn_cache: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute encoded features.

        Args:
            x (torch.Tensor): (#batch, time, size)
            mask (torch.Tensor): Mask tensor for the input (#batch, time, time).
            pos_emb (torch.Tensor): positional encoding, must not be None
                for ConformerEncoderLayer.
            mask_pad (torch.Tensor): batch padding mask used for conv module.
                (#batch, 1, time)
            attn_cache (torch.Tensor): Cache tensor of the output
                (#batch, time2, size), time2 < time in x.
            cnn_cache (torch.Tensor): Convolution cache in conformer layer
        Returns:
            torch.Tensor: Output tensor (#batch, time, size).
            torch.Tensor: Mask tensor (#batch, time).
        """
        #######################
        ## macaron feed forward
        #######################
        # whether to use macaron style
        if self.macaron_feed_forward is not None:
            residual = x
            if self.prenorm:
                x = self.macaron_norm(x)
            x = residual + self.macaron_scale * self.dropout(
                self.macaron_feed_forward(x))
            if not self.prenorm:
                x = self.macaron_norm(x)

        ##################
        ## self attention
        ##################
        residual = x
        if self.prenorm:
            x = self.attn_norm(x)

        attn_cache_is_none = (attn_cache is None or attn_cache.size(0) == 0)

        if attn_cache_is_none:
            x_q = x
        else:
            assert attn_cache.size(0) == x.size(0)
            assert attn_cache.size(2) == self.dim
            assert attn_cache.size(1) < x.size(1)
            offset = x.size(1) - attn_cache.size(1)
            x_q = x[:, -offset:, :]
            residual = residual[:, -offset:, :]
            mask = mask[:, -offset:, :]

        x_att = self.self_attn(x_q, x, x, mask, pos_emb)
        if self.concat_after:
            x_concat = torch.cat((x, x_att), dim=-1)
            x = residual + self.concat_linear(x_concat)
        else:
            x = residual + self.dropout(x_att)
        if not self.prenorm:
            x = self.attn_norm(x)

        #####################
        ## convolution module
        #####################
        # Fake new cnn cache here, and then change it in conv_module
        new_cnn_cache = torch.tensor([0.0], dtype=x.dtype, device=x.device)
        if self.conv_module is not None:
            residual = x
            if self.prenorm:
                x = self.conv_norm(x)
            x, new_cnn_cache = self.conv_module(x, mask_pad, cnn_cache)
            x = residual + self.dropout(x)
            if not self.prenorm:
                x = self.conv_norm(x)

        ###############
        ## feed forward
        ###############
        residual = x
        if self.prenorm:
            x = self.feed_norm(x)
        x = residual + self.macaron_scale * self.feed_forward(x)
        if not self.prenorm:
            x = self.feed_norm(x)

        ################
        ## final process
        ################
        if self.conv_module is not None:
            x = self.final_norm(x)

        if not attn_cache_is_none:
            x = torch.cat([attn_cache, x], dim=1)

        new_attn_cache = x
        return x, mask, new_attn_cache, new_cnn_cache

