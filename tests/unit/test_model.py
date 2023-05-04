import os
import pytest
import torch

from mountaintop.utils.yaml import load_yaml
from mountaintop.models.asr.transformer import AsrTransformer
from mountaintop.models.asr.utils import EncoderMode

root_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
toydata_dir = os.path.join(root_dir, "toydata")
config_path = os.path.join(toydata_dir, "conf/transformer.yaml")


@pytest.mark.parametrize(
    'batch_size', [20, 30]
)
@pytest.mark.parametrize(
    'input_len', [128, 256]
)
@pytest.mark.parametrize(
    'target_len', [8, 16]
)
@pytest.mark.parametrize(
    'num_classes', [100, 200]
)
def test_model_loss_valid(batch_size, input_len, target_len, num_classes):
    config = load_yaml(config_path)
    model_config = config["model"]
    model = AsrTransformer.create_from_config(model_config)
    model.eval()
    
    feat_dim = model_config["base_config"]["embed_size"]
    
    feat = torch.rand((batch_size, input_len, feat_dim))
    feat_lengths = torch.randint(low=input_len//2, high=input_len-10, size=(batch_size,))
    text = torch.randint(low=0, high=num_classes, size=(batch_size, target_len))
    text_lengths = torch.randint(low=1, high=target_len, size=(batch_size,))
    
    predictions, loss_total, metrics = model(feat, feat_lengths, text, text_lengths)
    for key, value in metrics.items():
        assert 0.0 < value < float("inf"), f"value[{key}] not valid, should in (0, inf)"
    assert 0.0 < loss_total < float("inf"), "total loss not valid"


@pytest.mark.parametrize(
    'input_len', [128, 256]
)
@pytest.mark.parametrize(
    'chunk_size', [1]
)
@pytest.mark.parametrize(
    'num_left_chunks', [2]
)
@pytest.mark.parametrize(
    'mode', ["offline", "dynamic_chunk", "static_chunk", "stream"]
)
def test_forward_encoder(input_len, chunk_size, num_left_chunks, mode):
    config = load_yaml(config_path)
    model_config = config["model"]
    if mode == "stream":
        model_config["encoder"]["use_dynamic_chunk"] = True
    model = AsrTransformer.create_from_config(model_config)
    model.eval()
    
    batch_size = 1 if mode == "stream" else 10
    feat_dim = model_config["base_config"]["embed_size"]
    hidden_size = model_config["base_config"]["hidden_size"]
    
    feat = torch.rand((batch_size, input_len, feat_dim))
    feat_lengths = torch.randint(low=input_len//2, high=input_len-10, size=(batch_size,))
    encoder_mode = EncoderMode.to_enum(mode)

    encoder_out, encoder_mask = model._internal_forward_encoder(
        feat=feat, 
        feat_lengths=feat_lengths, 
        mode=encoder_mode,
        chunk_size=chunk_size, 
        num_left_chunks=num_left_chunks, 
    )
    assert encoder_out.size() == (batch_size, (input_len-3)//model.subsampling_rate(), hidden_size)
    assert encoder_mask.size() == (batch_size, 1, (input_len-3)//model.subsampling_rate())


@pytest.mark.parametrize(
    'input_len', [128, 256]
)
@pytest.mark.parametrize(
    'chunk_size', [1]
)
@pytest.mark.parametrize(
    'num_left_chunks', [2]
)
@pytest.mark.parametrize(
    'beam_size', [10]
)
@pytest.mark.parametrize(
    'ctc_weight', [0.5]
)
@pytest.mark.parametrize(
    'mode', ["offline", "dynamic_chunk", "static_chunk", "stream"]
)
def test_attention_rescore(input_len, chunk_size, num_left_chunks, beam_size, ctc_weight, mode):
    config = load_yaml(config_path)
    model_config = config["model"]
    model_config["encoder"]["use_dynamic_chunk"] = True
    model = AsrTransformer.create_from_config(model_config)
    model.eval()
    
    batch_size = 1
    feat_dim = model_config["base_config"]["embed_size"]
    feat = torch.rand((batch_size, input_len, feat_dim))
    feat_lengths = torch.randint(low=input_len//2, high=input_len-10, size=(batch_size,))

    result = model.decode_by_rescore(
        feat=feat, 
        feat_lengths=feat_lengths, 
        mode=mode,
        beam_size=beam_size, 
        chunk_size=chunk_size, 
        num_left_chunks=num_left_chunks, 
        ctc_weight=ctc_weight, 
    )


@pytest.mark.parametrize(
    'batch_size', [20, 30]
)
@pytest.mark.parametrize(
    'input_len', [128, 256]
)
@pytest.mark.parametrize(
    'chunk_size', [1]
)
@pytest.mark.parametrize(
    'num_left_chunks', [2]
)
@pytest.mark.parametrize(
    'mode', ["offline", "dynamic_chunk", "static_chunk", "stream"]
)
def test_ctc_greedy_search(batch_size, input_len, chunk_size, num_left_chunks, mode):
    config = load_yaml(config_path)
    model_config = config["model"]
    if mode == "stream":
        model_config["encoder"]["use_dynamic_chunk"] = True
        batch_size = 1
    model = AsrTransformer.create_from_config(model_config)
    model.eval()
    
    feat_dim = model_config["base_config"]["embed_size"]
    feat = torch.rand((batch_size, input_len, feat_dim))
    feat_lengths = torch.randint(low=input_len//2, high=input_len-10, size=(batch_size,))
    
    hyps = model.decode_by_ctc_greedy(
        feat=feat, 
        feat_lengths=feat_lengths, 
        mode=mode,
        chunk_size=chunk_size, 
        num_left_chunks=num_left_chunks, 
    )


@pytest.mark.parametrize(
    'input_len', [128, 256]
)
@pytest.mark.parametrize(
    'target_len', [8, 16]
)
@pytest.mark.parametrize(
    'num_classes', [100, 200]
)
def test_forward_attention_decoder(input_len, target_len, num_classes):
    config = load_yaml(config_path)
    model_config = config["model"]
    model = AsrTransformer.create_from_config(model_config)
    model.eval()
    
    hidden_size = model_config["base_config"]["hidden_size"]

    batch_size = 1
    memory = torch.rand((batch_size, input_len, hidden_size))
    text = torch.randint(low=0, high=num_classes, size=(batch_size, target_len))
    text_lengths = torch.randint(low=1, high=target_len, size=(batch_size,))
    
    left, right = model.forward_attention_decoder(
        hyps=text,
        hyps_lens=text_lengths, 
        encoder_out=memory, 
        use_right_decoder=False
    )


@pytest.mark.parametrize(
    'batch_size', [20, 30]
)
@pytest.mark.parametrize(
    'input_len', [128, 256]
)
@pytest.mark.parametrize(
    'chunk_size', [1]
)
@pytest.mark.parametrize(
    'num_left_chunks', [2]
)
@pytest.mark.parametrize(
    'beam_size', [10]
)
@pytest.mark.parametrize(
    'mode', ["offline", "dynamic_chunk", "static_chunk", "stream"]
)
def test_attention_decode(batch_size, input_len, chunk_size, num_left_chunks, beam_size, mode):
    config = load_yaml(config_path)
    model_config = config["model"]
    if mode == "stream":
        model_config["encoder"]["use_dynamic_chunk"] = True
        batch_size = 1
    model = AsrTransformer.create_from_config(model_config)
    model.eval()
    
    feat_dim = model_config["base_config"]["embed_size"]
    feat = torch.rand((batch_size, input_len, feat_dim))
    feat_lengths = torch.randint(low=input_len//2, high=input_len-10, size=(batch_size,))
    
    best_hyps = model.decode_by_attention(
        feat=feat, 
        feat_lengths=feat_lengths, 
        mode=mode,
        beam_size=beam_size, 
        chunk_size=chunk_size, 
        num_left_chunks=num_left_chunks, 
    )
    assert best_hyps.size(0) == batch_size


