import torch


from mountaintop.layers.transformer.embedding import (
    NoPositionalEncoding, 
    PositionalEncoding, 
    NoScalePositionalEncoding, 
    RelPositionalEncoding, 
    NoScaleRelPositionalEncoding
)
from mountaintop.layers.subsampling import (
    LinearNoSubsampling, 
    Conv2dSubsampling4, 
    Conv2dSubsampling6, 
    Conv2dSubsampling8
)
from mountaintop.layers.normalization import CMVN
from mountaintop.core.audio.cmvn import load_cmvn
from mountaintop.layers.transformer.k2_conformer import Conv2dSubsampling
from mountaintop.runx.logx import loggerx
from mountaintop.layers.base.interface import LayerInterface


class SpeechEmbed(LayerInterface):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        pos_type: str = "no_pos",
        subsampling_type: str = "linear", 
        dropout_rate: float = 0.0,
        positional_dropout_rate: float = 0.0, 
        cmvn_file: str = None, 
        cmvn_type: str = "json",
        **kwargs,
    ) -> None:
        super().__init__()
        
        pos_enc_dict = {
            "abs_pos": PositionalEncoding,
            "rel_pos": RelPositionalEncoding,
            "no_pos": NoPositionalEncoding,
            "noscale_abs_pos": NoScalePositionalEncoding,
            "noscale_rel_pos": NoScaleRelPositionalEncoding,
        }
        subsampling_dict = {
            "linear": LinearNoSubsampling,
            "conv2d4": Conv2dSubsampling4,
            "conv2d6": Conv2dSubsampling6,
            "conv2d8": Conv2dSubsampling8,
            "k2conv2d4": Conv2dSubsampling,
        }
        
        assert pos_type in pos_enc_dict
        assert subsampling_type in subsampling_dict
        
        self._subsampling = subsampling_dict[subsampling_type](in_dim, out_dim, dropout_rate)
        self._position_encoder = pos_enc_dict[pos_type](out_dim, positional_dropout_rate)
        
        self._cmvn = None
        mean, vars = None, None
        if cmvn_file:
            assert cmvn_type in ["json", "kaldi"]
            mean, vars = load_cmvn(cmvn_file, cmvn_type)
        
        if mean is not None and vars is not None:
            loggerx.debug("using pre-computing cmvn")
            self._cmvn = CMVN(torch.Tensor(mean), torch.Tensor(vars))
            self._transpose = False
        else:
            loggerx.warning("Not setting CMVN file. Using BatchNorm insted of cmvn")
            self._cmvn = torch.nn.BatchNorm1d(in_dim)
            self._transpose = True
            
    def forward(self, x, mask, offset: int=0):
        if self._cmvn is not None:
            if self._transpose:
                x = self._cmvn(x.transpose(1, 2)).transpose(1, 2)
            else:
                x = self._cmvn(x)
        x, mask = self._subsampling(x, mask)
        y, pos_emb = self._position_encoder(x, offset)
        return y, mask, pos_emb
    
    def position_encoding(self, offset: int, size: int):
        return self._position_encoder.position_encoding(offset, size)
    
    @property
    def subsampling_rate(self):
        return self._subsampling.rate
    
    @property
    def right_context(self):
        return self._subsampling.right_context
