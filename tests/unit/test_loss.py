import pytest
import torch


from mountaintop.layers.loss import (
    CTC, 
    LabelSmoothingKlLoss, 
    LabelSmoothingCeLoss
)


@pytest.mark.parametrize(
    'batch_size', [20, 30]
)
@pytest.mark.parametrize(
    'input_len', [128, 256]
)
@pytest.mark.parametrize(
    'dim', [128, 256]
)
@pytest.mark.parametrize(
    'target_len', [16, 32]
)
@pytest.mark.parametrize(
    'num_classes', [1000]
)
def test_ctc_valid(batch_size, input_len, dim, target_len, num_classes):
    ctc_config = {
        "in_dim": dim,
        "num_classes": num_classes,
        "dropout_rate": 0.0,
        "reduction": "mean",
    }
    ctc_mean = CTC(**ctc_config)
    input = torch.rand(size=(batch_size, input_len, dim))
    intput_lens = torch.full(size=(batch_size,), fill_value=input_len, dtype=torch.long)
    target = torch.randint(low=1, high=num_classes, size=(batch_size, target_len), dtype=torch.long)
    taget_lens = torch.randint(low=10, high=target_len, size=(batch_size,), dtype=torch.long)

    loss = ctc_mean(input, intput_lens, target, taget_lens)
    assert loss < float("inf") and loss > -float("inf"), "loss is INF"


@pytest.mark.parametrize(
    'batch_size', [20, 30]
)
@pytest.mark.parametrize(
    'input_len', [16, 32]
)
@pytest.mark.parametrize(
    'dim', [128, 256]
)
@pytest.mark.parametrize(
    'target_len', [128, 256]
)
@pytest.mark.parametrize(
    'num_classes', [1000]
)
def test_ctc_not_valid(batch_size, input_len, dim, target_len, num_classes):
    ctc_config = {
        "in_dim": dim,
        "num_classes": num_classes,
        "dropout_rate": 0.0,
        "reduction": "mean",
    }
    ctc_mean = CTC(**ctc_config)
    input = torch.rand(size=(batch_size, input_len, dim))
    intput_lens = torch.full(size=(batch_size,), fill_value=input_len, dtype=torch.long)
    target = torch.randint(low=1, high=num_classes, size=(batch_size, target_len), dtype=torch.long)
    taget_lens = torch.randint(low=10, high=target_len, size=(batch_size,), dtype=torch.long)

    loss = ctc_mean(input, intput_lens, target, taget_lens)
    assert loss == float("inf") or loss == -float("inf"), "loss is not INF"



@pytest.mark.parametrize(
    'batch_size', [20, 30]
)
@pytest.mark.parametrize(
    'input_len', [16, 32]
)
@pytest.mark.parametrize(
    'num_classes', [1000]
)
def test_klloss_valid(batch_size, input_len, num_classes):

    loss_config = {
        "smoothing": 0.1, 
        "reduction": "mean"
    }
    kl_loss = LabelSmoothingKlLoss(**loss_config)
    # kl_loss.eval()
    x = torch.rand(size=(batch_size, input_len, num_classes), dtype=torch.float32)
    tgt = torch.randint(size=(batch_size, input_len), low=0, high=num_classes)
    taget_lens = torch.randint(low=2, high=input_len, size=(batch_size,), dtype=torch.long)
    loss = kl_loss(x, tgt, taget_lens)
    assert loss < float("inf") and loss > -float("inf"), "loss is INF"


@pytest.mark.parametrize(
    'batch_size', [20, 30]
)
@pytest.mark.parametrize(
    'input_len', [16, 32]
)
@pytest.mark.parametrize(
    'num_classes', [1000]
)
def test_celoss_valid(batch_size, input_len, num_classes):

    loss_config = {
        "smoothing": 0.1, 
        "reduction": "mean"
    }
    kl_loss = LabelSmoothingCeLoss(**loss_config)
    # kl_loss.eval()
    x = torch.rand(size=(batch_size, input_len, num_classes), dtype=torch.float32)
    tgt = torch.randint(size=(batch_size, input_len), low=0, high=num_classes)
    taget_lens = torch.randint(low=2, high=input_len, size=(batch_size,), dtype=torch.long)
    loss = kl_loss(x, tgt, taget_lens)
    assert loss < float("inf") and loss > -float("inf"), "loss is INF"



