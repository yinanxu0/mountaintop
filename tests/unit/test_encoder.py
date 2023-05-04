import pytest
import torch


from mountaintop.layers.transformer.encoder import TransformerEncoder
from mountaintop.layers.transformer.encoder_block import (
    TransformerEncoderBlock, 
    ConformerEncoderBlock
)


CONFIG = {
    "num_heads": 4,
    "num_hidden": 1024,
    "num_blocks": 3,
    "dropout_rate": 0.0,
    "attention_dropout_rate": 0.0,
    "norm_type": "prenorm",
    "activation_name": "relu",
    "concat_after": False,
    "chunk_size": 0,
    "use_dynamic_chunk": False,
    ### params for conformer
    "use_macaron": True,
    "use_cnn": True,
    "cnn_kernels": 15,
    "cnn_norm": "batchnorm" ,
    "# cnn_norm": "layernorm" ,
    "cnn_causal": False,
    "cnn_activation_name": "swish",
}


@pytest.mark.parametrize(
    'batch_size', [20, 30]
)
@pytest.mark.parametrize(
    'input_len', [32, 64]
)
@pytest.mark.parametrize(
    'dim', [128, 256]
)
@pytest.mark.parametrize(
    'block', [
        TransformerEncoderBlock, 
        ConformerEncoderBlock
    ]
)
@pytest.mark.parametrize(
    'config', [CONFIG]
)
def test_encoder_block_forward(batch_size, input_len, dim, block, config):
    config["dim"] = dim
    encoder_layer = block(**config)
    encoder_layer.eval()
    config.pop("dim")
    
    x = torch.rand((batch_size, input_len, dim))
    mask = torch.randint(low=0, high=1, size=(batch_size, input_len, input_len)).to(torch.bool)
    pos_emb = torch.rand((batch_size, input_len*2, dim))
    mask_pad = torch.randint(low=0, high=1, size=(batch_size, 1, input_len)).to(torch.bool)
    
    for idx in range(input_len//2, input_len, 4):
        tmp_x = x[:, :idx, :]
        tmp_mask = mask[:, :idx, :idx]
        if isinstance(encoder_layer, ConformerEncoderBlock):
            tmp_pos_emb = pos_emb[:, :2*idx-1, :]
        else:
            tmp_pos_emb = pos_emb[:, :idx, :]
        tmp_mask_pad = mask_pad[:, :, :idx]
        y, _, _, _ = encoder_layer(tmp_x, tmp_mask, tmp_pos_emb, tmp_mask_pad)
        assert y.size() == (batch_size, idx, dim)


@pytest.mark.parametrize(
    'batch_size', [20, 30]
)
@pytest.mark.parametrize(
    'input_len', [32, 64]
)
@pytest.mark.parametrize(
    'dim', [128, 256]
)
@pytest.mark.parametrize(
    'block', [
        "mountaintop.layers.transformer.encoder_block:TransformerEncoderBlock", 
        "mountaintop.layers.transformer.encoder_block:ConformerEncoderBlock", 
    ]
)
@pytest.mark.parametrize(
    'config', [CONFIG]
)
def test_encoder_forward(batch_size, input_len, dim, block, config):
    config["dim"] = dim
    config["block_module"] = block
    encoder = TransformerEncoder(**config)
    encoder.eval()
    config.pop("dim")
    config.pop("block_module")
    
    x = torch.rand((batch_size, input_len, dim))
    mask = torch.randint(low=0, high=1, size=(batch_size, 1, input_len)).to(torch.bool)
    pos_emb = torch.rand((batch_size, input_len*2, dim))

    for idx in range(input_len//2, input_len, 4):
        tmp_x = x[:, :idx, :]
        tmp_mask = mask[:, :, :idx]
        if "ConformerEncoderBlock" in block:
            tmp_pos_emb = pos_emb[:, :2*idx-1, :]
        else:
            tmp_pos_emb = pos_emb[:, :idx, :]
        
        y, _ = encoder(tmp_x, tmp_mask, tmp_pos_emb)
        assert y.size() == (batch_size, idx, dim)



@pytest.mark.parametrize(
    'input_len', [32, 64]
)
@pytest.mark.parametrize(
    'dim', [128, 256]
)
@pytest.mark.parametrize(
    'chunk_size', [1, 3]
)
@pytest.mark.parametrize(
    'left_num_chunks', [1, 3, 5]
)
@pytest.mark.parametrize(
    'block', [
        "mountaintop.layers.transformer.encoder_block:TransformerEncoderBlock", 
        "mountaintop.layers.transformer.encoder_block:ConformerEncoderBlock", 
    ]
)
@pytest.mark.parametrize(
    'config', [CONFIG]
)
def test_encoder_forward_one_chunk(input_len, dim, chunk_size, left_num_chunks, block, config):
    config["dim"] = dim
    config["block_module"] = block
    encoder = TransformerEncoder(**config)
    encoder.eval()
    config.pop("dim")
    config.pop("block_module")
    
    batch_size = 1
    attn_cache = []
    cnn_cache = []
    x = torch.rand((batch_size, input_len, dim))
    
    # mask = torch.randint(low=0, high=1, size=(batch_size, 1, input_len)).to(torch.bool)
    pos_emb = torch.rand((batch_size, input_len*2, dim))

    for idx in range(chunk_size, input_len, chunk_size):
        if left_num_chunks > 0 :
            if idx < (chunk_size+left_num_chunks):
                continue
            start = max(0, idx-(chunk_size+left_num_chunks))
            tmp_x = x[:, start:idx, :]
        else:
            tmp_x = x[:, :idx, :]
        if "ConformerEncoderBlock" in block:
            length = tmp_x.size(1)
            tmp_pos_emb = pos_emb[:, :2*length-1, :]
        else:
            tmp_pos_emb = pos_emb[:, :idx, :]
        
        y, attn_cache, cnn_cache = encoder.forward_one_chunk(tmp_x, tmp_pos_emb, chunk_size, left_num_chunks, attn_cache, cnn_cache)
        assert y.size() == (batch_size, chunk_size, dim)
        


@pytest.mark.parametrize(
    'input_len', [32, 64]
)
@pytest.mark.parametrize(
    'dim', [128, 256]
)
@pytest.mark.parametrize(
    'chunk_size', [1, 3]
)
@pytest.mark.parametrize(
    'left_num_chunks', [1, 3, 5]
)
@pytest.mark.parametrize(
    'block', [
        "mountaintop.layers.transformer.encoder_block:TransformerEncoderBlock", 
        "mountaintop.layers.transformer.encoder_block:ConformerEncoderBlock", 
    ]
)
@pytest.mark.parametrize(
    'config', [CONFIG]
)
def test_encoder_forward_stream(input_len, dim, chunk_size, left_num_chunks, block, config):
    
    config["dim"] = dim
    config["block_module"] = block
    config["use_dynamic_chunk"] = True
    encoder = TransformerEncoder(**config)
    encoder.eval()
    config.pop("dim")
    config.pop("block_module")
    config.pop("use_dynamic_chunk")
    
    batch_size = 1
    x = torch.rand((batch_size, input_len, dim))
    
    # mask = torch.randint(low=0, high=1, size=(batch_size, 1, input_len)).to(torch.bool)
    pos_emb = torch.rand((batch_size, input_len*2, dim))

    y, _ = encoder.forward_stream(x, pos_emb, chunk_size, left_num_chunks)
    length = input_len - input_len%chunk_size
    assert y.size() == (batch_size, length, dim)

