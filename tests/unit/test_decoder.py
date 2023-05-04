import pytest
import torch


from mountaintop.layers.transformer.decoder import (
    TransformerDecoder, 
    BiTransformerDecoder
)
from mountaintop.layers.transformer.decoder_block import (
    TransformerDecoderBlock, 
    GPTDecoderBlock
)

CONFIG = {
    "num_heads": 4,
    "num_hidden": 1024,
    "num_blocks": 3,
    "dropout_rate": 0.0,
    "positional_dropout_rate": 0.0,
    "self_attention_dropout_rate": 0.0,
    "src_attention_dropout_rate": 0.0,
    "norm_type": 'prenorm',
    "concat_after": False,
    "use_output_layer": True,
}



@pytest.mark.parametrize(
    'batch_size', [20, 30]
)
@pytest.mark.parametrize(
    'seq_len', [16, 24]
)
@pytest.mark.parametrize(
    'dim', [128, 256]
)
@pytest.mark.parametrize(
    'vocab_size', [1000]
)
@pytest.mark.parametrize(
    'block', [
        TransformerDecoderBlock, 
        GPTDecoderBlock
    ]
)
@pytest.mark.parametrize(
    'config', [CONFIG]
)
def test_decoder_block_forward(batch_size, seq_len, dim, vocab_size, block, config):
    config["vocab_size"] = vocab_size
    config["in_dim"] = dim
    decoder_layer = block(**config)
    decoder_layer.eval()
    config.pop("vocab_size")
    config.pop("in_dim")
    
    # token embedding
    tgt = torch.rand((batch_size, seq_len, dim))
    tgt_mask = torch.rand((batch_size, 1, seq_len))
    # memory
    factor = 2
    feat_len = int(seq_len * factor)
    memory = torch.rand((batch_size, feat_len, dim))
    memory_mask = torch.rand((batch_size, 1, feat_len))
    
    cache = None
    for idx in range(1, seq_len):

        tmp_tgt = tgt[:, :idx, :]
        tmp_tgt_mask = tgt_mask[:, :, :idx]
        tmp_memory = memory[:, :idx*factor, :]
        tmp_memory_mask = memory_mask[:, :, :idx*factor]
        
        y, _, _, _ = decoder_layer(tmp_tgt, tmp_tgt_mask, tmp_memory, tmp_memory_mask, cache)
        assert y.size() == (batch_size, idx, dim)
        cache = y


@pytest.mark.parametrize(
    'batch_size', [20, 30]
)
@pytest.mark.parametrize(
    'seq_len', [16, 24]
)
@pytest.mark.parametrize(
    'dim', [128, 256]
)
@pytest.mark.parametrize(
    'vocab_size', [1000]
)
@pytest.mark.parametrize(
    'layer', [
        TransformerDecoder, 
        BiTransformerDecoder 
    ]
)
@pytest.mark.parametrize(
    'block', [
        "mountaintop.layers.transformer.decoder_block:TransformerDecoderBlock", 
        "mountaintop.layers.transformer.decoder_block:GPTDecoderBlock", 
    ]
)
@pytest.mark.parametrize(
    'config', [CONFIG]
)
def test_decoder_forward(batch_size, seq_len, dim, vocab_size, layer, block, config):
    config["vocab_size"] = vocab_size
    config["in_dim"] = dim
    config["block_module"] = block
    decoder = layer(**config)
    decoder.eval()
    config.pop("vocab_size")
    config.pop("in_dim")
    config.pop("block_module")
    
    token_idxs = torch.randint(low=0, high=vocab_size, size=(batch_size, seq_len))
    token_lens = torch.randint(low=1, high=seq_len, size=(batch_size,))
    for idx, token_len in enumerate(token_lens):
        token_idxs[idx][token_len:] = 0

    # memory
    factor = 2
    feat_len = int(seq_len * factor)
    memory = torch.rand((batch_size, feat_len, dim))
    memory_mask = torch.rand((batch_size, 1, feat_len))
    
    y, _, _ = decoder(memory, memory_mask, token_idxs, token_lens)
    assert y.size() == (batch_size, seq_len, vocab_size)


@pytest.mark.parametrize(
    'batch_size', [20, 30]
)
@pytest.mark.parametrize(
    'seq_len', [16, 24]
)
@pytest.mark.parametrize(
    'dim', [128, 256]
)
@pytest.mark.parametrize(
    'vocab_size', [1000]
)
@pytest.mark.parametrize(
    'layer', [
        TransformerDecoder, 
        BiTransformerDecoder 
    ]
)
@pytest.mark.parametrize(
    'block', [
        "mountaintop.layers.transformer.decoder_block:TransformerDecoderBlock", 
        "mountaintop.layers.transformer.decoder_block:GPTDecoderBlock", 
    ]
)
@pytest.mark.parametrize(
    'config', [CONFIG]
)
def test_decoder_forward_one_step(batch_size, seq_len, dim, vocab_size, layer, block, config):
    config["vocab_size"] = vocab_size
    config["in_dim"] = dim
    config["block_module"] = block
    decoder = layer(**config)
    decoder.eval()
    config.pop("vocab_size")
    config.pop("in_dim")
    config.pop("block_module")
    
    token_idxs = torch.randint(low=0, high=vocab_size, size=(batch_size, seq_len))
    token_lens = torch.randint(low=1, high=seq_len, size=(batch_size,))
    for idx, token_len in enumerate(token_lens):
        token_idxs[idx][token_len:] = 0

    # memory
    factor = 2
    feat_len = int(seq_len * factor)
    memory = torch.rand((batch_size, feat_len, dim))
    memory_mask = torch.rand((batch_size, 1, feat_len))
    
    cache = None
    for idx in range(100, seq_len):
        tgt = token_idxs[:, :idx]
        tgt_mask = torch.rand((batch_size, idx, idx))
        y, cache = decoder.forward_one_step(memory, memory_mask, tgt, tgt_mask, cache)
        assert y.size() == (batch_size, vocab_size)


