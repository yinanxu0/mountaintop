from typing import Tuple, List, Optional
import torch


from mountaintop.layers.transformer.embedding import PositionalEncoding
from mountaintop.layers.low_rank_transformer.low_rank_decoder_block import (
    LowRankTransformerDecoderBlock
)
from mountaintop.core.internal.module import import_module
from mountaintop.core.ops.mask import (subsequent_mask, make_valid_mask)
from mountaintop.core.ops.common import flip_tensor
from mountaintop.layers.base.interface import LayerInterface


class LowRankTransformerDecoder(LayerInterface):
    """Base class of Transfomer decoder module.
    Args:
        vocab_size: output dim
        in_dim: dimension of attention
        num_heads: the number of heads of multi head attention
        num_hidden: the hidden units number of position-wise feedforward
        num_blocks: the number of decoder blocks
        dropout_rate: dropout rate
        self_attention_dropout_rate: dropout rate for attention
        input_layer: input layer type
        use_output_layer: whether to use output layer
        pos_enc_class: PositionalEncoding or ScaledPositionalEncoding
        normalize_before:
            True: use layer_norm before each sub-block of a layer.
            False: use layer_norm after each sub-block of a layer.
        concat_after: whether to concat attention layer's input and output
            True: x -> x + linear(concat(x, att(x)))
            False: x -> x + att(x)
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
        # assert check_argument_types()
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
        
        Decoder_Block = LowRankTransformerDecoderBlock
        if kwargs.get("block_module", None) is not None:
            Decoder_Block = import_module(kwargs["block_module"])

        self.decoders = torch.nn.ModuleList([
            Decoder_Block(
                in_dim=in_dim, num_heads=num_heads, 
                num_hidden=num_hidden, dropout_rate=dropout_rate, 
                self_attention_dropout_rate=self_attention_dropout_rate,
                src_attention_dropout_rate=src_attention_dropout_rate,
                norm_type=norm_type, activation_name=activation_name, 
                concat_after=concat_after,
            ) for _ in range(num_blocks)
        ])

    def bind_embed(self):
        # TODO: embedding binding to the output layer
        ### setting embedding from pretrained weight
        # embedding = torch.nn.Embedding(vocab_size, in_dim) # init
        # embedding.weight = torch.nn.Parameter(emb) # load weight
        # embedding.weight.requires_grad = False # freeze embeddings
        pass
        
    def is_bidirectional(self):
        return False
    
    def forward(
        self,
        memory: torch.Tensor,
        memory_mask: torch.Tensor,
        tgt: torch.Tensor,
        tgt_lengths: torch.Tensor,
        use_right_decoder: bool = False,
        # *args, **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward decoder.
        Args:
            memory: encoded memory, float32  (batch, maxlen_in, feat)
            memory_mask: encoder memory mask, (batch, 1, maxlen_in)
            tgt: padded input token ids, int64 (batch, maxlen_out)
            tgt_lengths: input lengths of this batch (batch)
        Returns:
            (tuple): tuple containing:
                l_tgt_hat: decoded token score before softmax (batch, maxlen_out,
                    vocab_size) if use_output_layer is True,
                r_tgt_hat: x: decoded token score (right to left decoder)
                    before softmax (batch, maxlen_out, vocab_size)
                    if use_output_layer is True,
                olens: (batch, )
        """
        assert tgt_lengths.min().item() > 0
        assert tgt.dim() <= 3
        seqlen = tgt.size(1)
        # tgt_mask: (B, 1, L)
        tgt_mask = (make_valid_mask(tgt_lengths, maxlen=seqlen).unsqueeze(1)).to(tgt.device)
        # m: (1, L, L)
        m = subsequent_mask(tgt_mask.size(-1),
                            device=tgt_mask.device).unsqueeze(0)
        # tgt_mask: (B, L, L)
        tgt_mask = tgt_mask & m
        x, _ = self._embedding(tgt)
        for layer in self.decoders:
            x, tgt_mask, memory, memory_mask = layer(x, tgt_mask, memory, memory_mask)
        if self.prenorm:
            x = self._norm(x)
        if self.use_output_layer:
            x = self._output_layer(x)
        tgt_hat = x
        fake_tgt_hat = torch.zeros_like(tgt_hat)
        olens = tgt_mask.sum(1)
        
        return tgt_hat, fake_tgt_hat, olens

    def forward_one_step(
        self,
        memory: torch.Tensor,
        memory_mask: torch.Tensor,
        tgt: torch.Tensor,
        tgt_mask: torch.Tensor,
        cache: Optional[List[torch.Tensor]] = None,
        *args, **kwargs,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Forward one step.
            This is only used for decoding.
        Args:
            memory: encoded memory, float32  (batch, maxlen_in, feat)
            memory_mask: encoded memory mask, (batch, 1, maxlen_in)
            tgt: input token ids, int64 (batch, maxlen_out)
            tgt_mask: input token mask,  (batch, maxlen_out, maxlen_out)
                      dtype=torch.uint8 in PyTorch 1.2-
                      dtype=torch.bool in PyTorch 1.2+ (include 1.2)
            cache: cached output list of (batch, max_time_out-1, size)
        Returns:
            y, cache: NN output value and cache per `self.decoders`.
            y.shape` is (batch, maxlen_out, token)
        """
        x, _ = self._embedding(tgt)
        new_cache = []
        for idx, decoder in enumerate(self.decoders):
            c = None if cache is None else cache[idx]
            x, tgt_mask, memory, memory_mask = decoder(x, tgt_mask, memory, memory_mask, c)
            new_cache.append(x)
        
        y = x[:, -1]
        if self.prenorm:
            y = self._norm(y)
        if self.use_output_layer:
            y = torch.log_softmax(self._output_layer(y), dim=-1)
        return y, new_cache


class LowRankBiTransformerDecoder(LayerInterface):
    """Base class of Transfomer decoder module.
    Args:
        vocab_size: output dim
        in_dim: dimension of attention
        num_heads: the number of heads of multi head attention
        num_hidden: the hidden units number of position-wise feedforward
        num_blocks: the number of decoder blocks
        r_num_blocks: the number of right to left decoder blocks
        dropout_rate: dropout rate
        self_attention_dropout_rate: dropout rate for attention
        input_layer: input layer type
        use_output_layer: whether to use output layer
        pos_enc_class: PositionalEncoding or ScaledPositionalEncoding
        normalize_before:
            True: use layer_norm before each sub-block of a layer.
            False: use layer_norm after each sub-block of a layer.
        concat_after: whether to concat attention layer's input and output
            True: x -> x + linear(concat(x, att(x)))
            False: x -> x + att(x)
    """
    def __init__(
        self,
        vocab_size: int,
        in_dim: int,
        num_heads: int = 4,
        num_hidden: int = 2048,
        num_blocks: int = 3,
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
        # assert check_argument_types()
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

        self._left_norm = torch.nn.LayerNorm(in_dim, eps=1e-5) if self.prenorm else None
        self._right_norm = torch.nn.LayerNorm(in_dim, eps=1e-5) if self.prenorm else None
        self._output_layer = torch.nn.Linear(in_dim, vocab_size) if use_output_layer else None
        
        Decoder_Block = LowRankTransformerDecoderBlock
        if kwargs.get("block_module", None) is not None:
            Decoder_Block = import_module(kwargs["block_module"])
            
        self._left_decoders = torch.nn.ModuleList([
            Decoder_Block(
                in_dim=in_dim, num_heads=num_heads, 
                num_hidden=num_hidden, dropout_rate=dropout_rate, 
                self_attention_dropout_rate=self_attention_dropout_rate,
                src_attention_dropout_rate=src_attention_dropout_rate,
                norm_type=norm_type, activation_name=activation_name, 
                concat_after=concat_after,
            ) for _ in range(num_blocks)
        ])
        self._right_decoders = torch.nn.ModuleList([
            Decoder_Block(
                in_dim=in_dim, num_heads=num_heads, 
                num_hidden=num_hidden, dropout_rate=dropout_rate, 
                self_attention_dropout_rate=self_attention_dropout_rate,
                src_attention_dropout_rate=src_attention_dropout_rate,
                norm_type=norm_type, activation_name=activation_name, 
                concat_after=concat_after,
            ) for _ in range(num_blocks)
        ])

    def is_bidirectional(self):
        return True
    
    def forward(
        self,
        memory: torch.Tensor,
        memory_mask: torch.Tensor,
        tgt: torch.Tensor,
        tgt_lengths: torch.Tensor,
        use_right_decoder: bool = True,
        # *args, **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward decoder.
        Args:
            memory: encoded memory, float32  (batch, maxlen_in, feat)
            memory_mask: encoder memory mask, (batch, 1, maxlen_in)
            tgt: padded input token ids, int64 (batch, maxlen_out)
            tgt_lengths: input lengths of this batch (batch)
            use_right_decoder: use right to left decoder
        Returns:
            (tuple): tuple containing:
                l_tgt_hat: decoded token score before softmax (batch, maxlen_out,
                    vocab_size) if use_output_layer is True,
                r_tgt_hat: x: decoded token score (right to left decoder)
                    before softmax (batch, maxlen_out, vocab_size)
                    if use_output_layer is True,
                olens: (batch, )
        """
        assert tgt_lengths.min().item() > 0
        ### preprocess
        seqlen = tgt.size(1)
        # tgt_mask: (B, 1, L)
        tgt_mask = (make_valid_mask(tgt_lengths, maxlen=seqlen).unsqueeze(1)).to(tgt.device)
        # m: (1, L, L)
        temp_mask = subsequent_mask(tgt_mask.size(-1),
                            device=tgt_mask.device).unsqueeze(0)
        # tgt_mask: (B, L, L)
        tgt_mask = tgt_mask & temp_mask
        
        ### compute left2right
        x, _ = self._embedding(tgt)
        for left_decoder in self._left_decoders:
            x, tgt_mask, memory, memory_mask = left_decoder(x, tgt_mask, memory, memory_mask)
        if self.prenorm:
            x = self._left_norm(x)
        if self.use_output_layer:
            x = self._output_layer(x)
        
        l_tgt_hat = x
        r_tgt_hat = torch.zeros_like(l_tgt_hat)
        olens = tgt_mask.sum(1)
        ### compute right2left
        if use_right_decoder:
            reverse_tgt = flip_tensor(tgt=tgt, tgt_lengths=tgt_lengths, start=1)
            x, _ = self._embedding(reverse_tgt)
            for right_decoder in self._right_decoders:
                x, tgt_mask, memory, memory_mask = right_decoder(x, tgt_mask, memory, memory_mask)
            if self.prenorm:
                x = self._right_norm(x)
            if self.use_output_layer:
                x = self._output_layer(x)
            r_tgt_hat = x
        return l_tgt_hat, r_tgt_hat, olens

    def forward_one_step(
        self,
        memory: torch.Tensor,
        memory_mask: torch.Tensor,
        tgt: torch.Tensor,
        tgt_mask: torch.Tensor,
        cache: Optional[List[torch.Tensor]] = None,
        *args, **kwargs,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Forward one step.
            This is only used for decoding.
        Args:
            memory: encoded memory, float32  (batch, maxlen_in, feat)
            memory_mask: encoded memory mask, (batch, 1, maxlen_in)
            tgt: input token ids, int64 (batch, maxlen_out)
            tgt_mask: input token mask,  (batch, maxlen_out)
                      dtype=torch.uint8 in PyTorch 1.2-
                      dtype=torch.bool in PyTorch 1.2+ (include 1.2)
            cache: cached output list of (batch, max_time_out-1, size)
        Returns:
            y, cache: NN output value and cache per `self.decoders`.
            y.shape` is (batch, maxlen_out, token)
        """
        x, _ = self._embedding(tgt)
        new_cache = []
        for idx, left_decoder in enumerate(self._left_decoders):
            c = None if cache is None else cache[idx]
            x, tgt_mask, memory, memory_mask = left_decoder(x, tgt_mask, memory, memory_mask, c)
            new_cache.append(x)
        
        y = x[:, -1]
        if self.prenorm:
            y = self._left_norm(y)
        if self.use_output_layer:
            y = torch.log_softmax(self._output_layer(y), dim=-1)
        return y, new_cache
