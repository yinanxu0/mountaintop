from typing import Tuple, List, Optional
import torch


from mountaintop.core.ops.mask import add_chunk_mask_with_mode
from mountaintop.core.internal.module import import_module
from mountaintop.layers.low_rank_transformer.low_rank_encoder_block import (
    LowRankTransformerEncoderBlock,
    LowRankConformerEncoderBlock
)
from mountaintop.layers.base.interface import LayerInterface
from mountaintop.models.asr.utils import EncoderMode

class LowRankTransformerEncoder(LayerInterface):
    def __init__(
        self,
        dim: int = 256,
        num_heads: int = 4,
        num_hidden: int = 2048,
        num_blocks: int = 6,
        dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.0,
        norm_type: str = "prenorm",
        activation_name: str = 'relu',
        concat_after: bool = False,
        chunk_size: int = 0,
        use_dynamic_chunk: bool = False,
        use_macaron: bool = True,
        use_cnn: bool = True,
        cnn_kernels: int = 15,
        cnn_norm: str = "batchnorm", 
        cnn_causal: bool = True,
        cnn_activation_name: str = "swish", 
        *args, **kwargs,
    ):
        """
        Args:
            in_dim (int): input dim
            out_dim (int): dimension of attention
            num_heads (int): the number of heads of multi head attention
            num_hidden (int): the hidden units number of position-wise feed forward
            num_blocks (int): the number of decoder blocks
            dropout_rate (float): dropout rate
            attention_dropout_rate (float): dropout rate in attention
            positional_dropout_rate (float): dropout rate after adding
                positional encoding
            input_layer (str): input layer type.
                optional [linear, conv2d, conv2d6, conv2d8]
            pos_enc_layer_type (str): Encoder positional encoding layer type.
                opitonal [abs_pos, scaled_abs_pos, rel_pos, no_pos]
            norm_type (str): use pre-norm or post-norm before each sub-block 
                of a layer
            activation_name (str): activation in position feed forward layer
            concat_after (bool): whether to concat attention layer's input
                and output.
                True: x -> x + linear(concat(x, att(x)))
                False: x -> x + att(x)
            chunk_size (int): chunk size for static chunk training and
                decoding
            use_dynamic_chunk (bool): whether use dynamic chunk size for
                training or not, You can only use fixed chunk(chunk_size > 0)
                or dyanmic chunk size(use_dynamic_chunk = True)
        """
        super().__init__()
        assert dim > 0
        assert num_heads > 0
        assert num_hidden > 0
        assert num_blocks > 0
        assert 0.0 <= dropout_rate <= 1.0
        assert 0.0 <= attention_dropout_rate <= 1.0
        
        self._dim = dim
        self._num_blocks = num_blocks
        
        Encoder_Block = LowRankTransformerEncoderBlock
        self.encoder_name = "transformer"
        if kwargs.get("block_module", None) is not None:
            Encoder_Block = import_module(kwargs["block_module"])
            if "conformer" in kwargs["block_module"].lower():
                self.encoder_name = "conformer"
        else:
            if kwargs.get("block", None) == "conformer":
                Encoder_Block = LowRankConformerEncoderBlock
                self.encoder_name = "conformer"

        self._encoders = torch.nn.ModuleList([
            Encoder_Block(
                dim=dim, num_heads=num_heads, num_hidden=num_hidden, dropout_rate=dropout_rate, 
                attention_dropout_rate=attention_dropout_rate, norm_type=norm_type, 
                activation_name=activation_name, concat_after=concat_after, use_macaron=use_macaron, 
                use_cnn=use_cnn, cnn_kernels=cnn_kernels, cnn_norm=cnn_norm, cnn_causal=cnn_causal,
                cnn_activation_name=cnn_activation_name, 
                *args, **kwargs
            ) for _ in range(num_blocks)
        ])

        self._prenorm = torch.nn.LayerNorm(dim, eps=1e-5) if norm_type == "prenorm" else None
        
        self.static_chunk_size = chunk_size
        self.use_dynamic_chunk = use_dynamic_chunk

    @property
    def dim(self) -> int:
        return self._dim

    def forward(
        self,
        feat: torch.Tensor,
        masks: torch.Tensor,
        pos_emb: torch.Tensor,
        mode: EncoderMode = EncoderMode.Offline,
        chunk_size: int = 0,
        num_left_chunks: int = -1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Embed positions in tensor.

        Args:
            feat: padded input tensor (B, T, D)
            masks: torch.Tensor,
            pos_emb: torch.Tensor,
            chunk_size: decoding chunk size for dynamic chunk
                0: default for training, use random dynamic chunk.
                <0: for decoding, use full chunk.
                >0: for decoding, use fixed chunk size as set.
            num_left_chunks: number of left chunks, this is for decoding,
                the chunk size is chunk_size.
                    >=0: use num_left_chunks
                    <0: use all left chunks
        Returns:
            encoder output tensor feat, and subsampled masks
            feat: padded output tensor (B, T' ~= T/subsample_rate, D)
            masks: torch.Tensor batch padding mask after subsample
                (B, 1, T' ~= T/subsample_rate)
        """
        mask_pad = masks  # (B, 1, T/subsample_rate)
        # TODO: need change inpurt args
        maxlen = feat.size(1)
        chunk_masks = add_chunk_mask_with_mode(
            size=maxlen, 
            masks=masks, 
            mode=mode,
            chunk_size=chunk_size, 
            num_left_chunks=num_left_chunks, 
            device=feat.device
        )
        y = feat
        for layer in self._encoders:
            y, chunk_masks, _, _ = layer(y, chunk_masks, pos_emb, mask_pad)
        if self._prenorm is not None:
            y = self._prenorm(y)
        # Here we assume the mask is not changed in encoder layers, so just
        # return the masks before encoder layers, and the masks will be used
        # for cross attention with decoder later
        return y, masks

    def forward_one_chunk(
        self,
        feat: torch.Tensor,
        pos_emb: torch.Tensor,
        chunk_size: int = 1,
        num_left_chunks: int = -1,
        attn_cache: Optional[List[torch.Tensor]] = None,
        cnn_cache: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        """ Forward just one chunk

        Args:
            feat (torch.Tensor): chunk input #(1, num_left_chunks+chunk_size, dim)
            offset (int): current offset in encoder output time stamp
            num_left_chunks (int): cache size required for next chunk
                compuation
                >=0: actual cache size
                <0: means all history cache is required
            subsampling_cache (Optional[torch.Tensor]): subsampling cache
            attn_cache (Optional[List[torch.Tensor]]):
                transformer/conformer encoder layers output cache
            cnn_cache (Optional[List[torch.Tensor]]): conformer cnn cache

        Returns:
            torch.Tensor: output of current input feat
            torch.Tensor: subsampling cache required for next chunk computation
            List[torch.Tensor]: encoder layers output cache required for next
                chunk computation
            List[torch.Tensor]: conformer cnn cache

        """
        assert feat.dim() == 3
        assert feat.size(0) == 1
        assert feat.size(1) >= chunk_size

        if num_left_chunks >= 0:
            next_cache_start = max(feat.size(1) - num_left_chunks, 0)
            whole_cache_size = min(num_left_chunks + chunk_size, feat.size(1))
        else:
            next_cache_start = 0
            whole_cache_size = feat.size(1)
        
        # cache_len = feat.size(1)
        feat = feat[:, -whole_cache_size:, :]
        # TODO: How to make this operation of conformer position embedding more decent?
        # if isinstance(self._encoders[0], LowRankConformerEncoderBlock):
        if self.encoder_name == "conformer":
            pos_emb = pos_emb[:, -(2*whole_cache_size-1):, :]
        else:
            pos_emb = pos_emb[:, -whole_cache_size:, :]
        masks = torch.ones(size=(1, whole_cache_size, whole_cache_size), device=feat.device, dtype=torch.bool)
        
        # run encoder layers
        new_attn_cache = []
        new_cnn_cache = []
        attn_cache_i = torch.zeros(size=(0, 0, 0, 0))
        cnn_cache_i = torch.zeros(size=(0, 0, 0, 0))
        for i, layer in enumerate(self._encoders):
            if attn_cache is not None and len(attn_cache) == self._num_blocks:
                attn_cache_i = attn_cache[i]
            if cnn_cache is not None and len(cnn_cache) == self._num_blocks:
                cnn_cache_i = cnn_cache[i]
            feat, _, new_attn_cache_i, new_cnn_cache_i = layer(feat, masks, pos_emb,
                                         attn_cache=attn_cache_i,
                                         cnn_cache=cnn_cache_i)
            new_attn_cache.append(new_attn_cache_i[:, next_cache_start:, :])
            new_cnn_cache.append(new_cnn_cache_i)
        
        if self._prenorm is not None:
            feat = self._prenorm(feat)

        return feat[:, -chunk_size:, :], new_attn_cache, new_cnn_cache

    def forward_stream(
        self,
        feat: torch.Tensor,
        pos_emb: torch.Tensor,
        chunk_size: int = 1,
        num_left_chunks: int = -1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Forward input chunk by chunk with chunk_size like a streaming
            fashion

        Here we should pay special attention to computation cache in the
        streaming style forward chunk by chunk. Three things should be taken
        into account for computation in the current network:
            1. transformer/conformer encoder layers output cache
            2. convolution in conformer
            3. convolution in subsampling

        However, we don't implement subsampling cache for:
            1. We can control subsampling module to output the right result by
               overlapping input instead of cache left context, even though it
               wastes some computation, but subsampling only takes a very
               small fraction of computation in the whole model.
            2. Typically, there are several covolution layers with subsampling
               in subsampling module, it is tricky and complicated to do cache
               with different convolution layers with different subsampling
               rate.
            3. Currently, nn.Sequential is used to stack all the convolution
               layers in subsampling, we need to rewrite it to make it work
               with cache, which is not prefered.
        Args:
            feat (torch.Tensor): (1, max_len, dim)
            chunk_size (int): decoding chunk size
        """
        #################### following is new implementation
        assert chunk_size > 0
        # The model is trained by static or dynamic chunk
        assert self.static_chunk_size > 0 or self.use_dynamic_chunk
        
        num_frames = feat.size(1)
        cache_size = chunk_size + num_left_chunks
        
        outputs = []
        attn_cache = []
        cnn_cache = []
        #TODO: padding 0 to the left of feat?
        # Feed forward overlap input step by step
        # TODO: when num_frames%chunk_size != 0, how to process remain input?
        for idx in range(chunk_size, num_frames+1, chunk_size):
            ####### JIT version
            if num_left_chunks > 0 :
                start = max(idx-cache_size, 0)
            else:
                start = 0
            chunk_feat = feat[:, start:idx, :]
            if self.encoder_name == "conformer":
                chunk_pos_emb = pos_emb[:, 2*start:2*idx, :]
            else:
                chunk_pos_emb = pos_emb[:, start:idx, :]
            
            (y, attn_cache, cnn_cache) = self.forward_one_chunk(
                                            chunk_feat, chunk_pos_emb, chunk_size,
                                            num_left_chunks, attn_cache, cnn_cache)
            outputs.append(y)
        y = torch.cat(outputs, 1)
        masks = torch.ones(size=(1, 1, y.size(1)), device=y.device, dtype=torch.bool)
        return y, masks
