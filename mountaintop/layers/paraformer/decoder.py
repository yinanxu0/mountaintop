from typing import Tuple, List, Optional
import torch
from typeguard import check_argument_types


from mountaintop.layers.transformer.embedding import PositionalEncoding
from mountaintop.layers.transformer.decoder_block import (
    TransformerDecoderBlock, 
)
from mountaintop.layers.base.interface import LayerInterface
from mountaintop.core.ops.mask import make_valid_mask


class ParaformerDecoder(LayerInterface):
    """
    Paraformer: Fast and Accurate Parallel Transformer for Non-autoregressive 
    End-to-End Speech Recognition
    https://arxiv.org/abs/2006.01713
    """
    def __init__(
        self,
        vocab_size: int,
        in_dim: int,
        num_heads: int = 4,
        num_hidden: int = 2048,
        num_blocks: int = 6,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        self_attention_dropout_rate: float = 0.0,
        src_attention_dropout_rate: float = 0.0,
        norm_type: str = "prenorm", 
        activation_name: str = 'relu',
        use_output_layer: bool = True,
        concat_after: bool = False,
        *args, **kwargs,
    ):
        assert check_argument_types()
        super().__init__()
        assert norm_type in ["prenorm", "postnorm"]
        assert in_dim > 0
        assert num_heads > 0
        assert num_hidden > 0
        assert num_blocks > 0
        
        self.in_dim = in_dim
        self.prenorm = norm_type == "prenorm"
        self.concat_after = concat_after
        self.use_output_layer = use_output_layer
        
        self._embedding = torch.nn.Sequential(
            torch.nn.Embedding(vocab_size, in_dim),
            PositionalEncoding(in_dim, positional_dropout_rate),
        )

        self._norm = torch.nn.LayerNorm(in_dim, eps=1e-5) if self.prenorm else None
        self._output_layer = torch.nn.Linear(in_dim, vocab_size) if use_output_layer else None
        
        Decoder_Block = TransformerDecoderBlock
        self._decoders = torch.nn.ModuleList([
            Decoder_Block(
                in_dim=in_dim, num_heads=num_heads, 
                num_hidden=num_hidden, dropout_rate=dropout_rate, 
                self_attention_dropout_rate=self_attention_dropout_rate,
                src_attention_dropout_rate=src_attention_dropout_rate,
                norm_type=norm_type, activation_name=activation_name, 
                concat_after=concat_after,
            ) for _ in range(num_blocks)
        ])

    def forward(
        self,
        memory: torch.Tensor,
        memory_mask: torch.Tensor,
        tgt: torch.Tensor,
        tgt_lengths: torch.Tensor,
        use_right_decoder: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward decoder.

        Args:
            memory: encoded memory, float32  (batch, maxlen_in, feat)
            hlens: (batch)
            tgt:
                input token ids, int64 (batch, maxlen_out)
                if input_layer == "embed"
                input tensor (batch, maxlen_out, #mels) in the other cases
            tgt_lengths: (batch)
        Returns:
            (tuple): tuple containing:

            x: decoded token score before softmax (batch, maxlen_out, token)
                if use_output_layer is True,
            olens: (batch, )
        """
        device = tgt.device
        seqlen = tgt.size(1)
        tgt_mask = (make_valid_mask(tgt_lengths, maxlen=seqlen).unsqueeze(1)).to(device)

        x, _ = self._embedding(tgt)
        for layer in self._decoders:
            x, tgt_mask, memory, memory_mask = layer(x, tgt_mask, memory, memory_mask)
        if self.prenorm:
            x = self._norm(x)
        if self.use_output_layer:
            x = self._output_layer(x)
        tgt_hat = x
        fake_tgt_hat = torch.zeros_like(tgt_hat)
        olens = tgt_mask.sum(1)
        return tgt_hat, fake_tgt_hat, olens

    def forward_by_embedding(
        self,
        memory: torch.Tensor,
        memory_mask: torch.Tensor,
        tgt_embedding: torch.Tensor,
        tgt_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward decoder.

        Args:
            memory: encoded memory, float32  (batch, maxlen_in, feat)
            hlens: (batch)
            tgt:
                input token ids, int64 (batch, maxlen_out)
                if input_layer == "embed"
                input tensor (batch, maxlen_out, #mels) in the other cases
            tgt_lengths: (batch)
        Returns:
            (tuple): tuple containing:

            x: decoded token score before softmax (batch, maxlen_out, token)
                if use_output_layer is True,
            olens: (batch, )
        """
        # tgt = tgt
        device = tgt_embedding.device
        seqlen = tgt_embedding.size(1)
        # tgt_mask = (~make_pad_mask(tgt_lengths)[:, None, :]).to(tgt.device)
        # tgt_mask: (B, 1, L)
        tgt_mask = (make_valid_mask(tgt_lengths, maxlen=seqlen).unsqueeze(1)).to(device)

        x = tgt_embedding
        for layer in self._decoders:
            x, tgt_mask, memory, memory_mask = layer(x, tgt_mask, memory, memory_mask)
        if self.prenorm:
            x = self._norm(x)
        if self.use_output_layer:
            x = self._output_layer(x)
        tgt_hat = x
        fake_tgt_hat = torch.zeros_like(tgt_hat)
        olens = tgt_mask.sum(1)
        return tgt_hat, fake_tgt_hat, olens
    
    def embed(self, tgt):
        return self._embedding(tgt)
        
    def forward_one_step(self):
        pass
    